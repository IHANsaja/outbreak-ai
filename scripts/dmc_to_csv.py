#!/usr/bin/env python3
"""
DMC report scraper -> CSV

What it does
1) Scrapes the DMC report listing page.
2) Downloads each report file (PDF or CSV).
3) Parses water-level PDFs into row-based records.
4) Falls back to raw-page records for other report formats.
5) Writes one combined CSV.

The script is designed around the DMC River Water Level report list
and the structure visible in the sample PDF you shared:
- page 1: river water level table
- page 2: rainfall / station rainfall table

Dependencies
    pip install requests beautifulsoup4 pdfplumber pandas

Optional (not required):
    pip install python-dateutil

Usage
    python dmc_to_csv.py --output dmc_reports.csv
    python dmc_to_csv.py --report-type-id 6 --from-date 2025-12-01 --to-date 2026-03-23
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import io
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib.parse import urljoin

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.dmc.gov.lk"
DEFAULT_LIST_URL = (
    BASE_URL
    + "/index.php?Itemid=277&lang=en&option=com_dmcreports"
    + "&report_type_id={report_type_id}&view=reports&limit=0"
)

# A practical list of station names seen in DMC river-water-level PDFs.
# The parser uses these to split "prefix" text into station vs. tributary.
KNOWN_STATIONS = sorted(
    {
        "Nagalagam Street",
        "Hanwella",
        "Glencourse",
        "Kithulgala",
        "Holombuwa",
        "Deraniyagala",
        "Norwood",
        "Putupaula",
        "Ellagawa",
        "Rathnapura",
        "Magura",
        "Kalawellawa (Millakanda)",
        "Baddegama",
        "Thawalama",
        "Thalgahagoda",
        "Panadugama",
        "Pitabeddara",
        "Urawa",
        "Moraketiya",
        "Thanamalwila",
        "Wellawaya",
        "Kuda Oya",
        "Katharagama",
        "Nakkala",
        "Siyambalanduwa",
        "Padiyathalawa",
        "Manampitiya",
        "Weraganthota",
        "Peradeniya",
        "Nawalapitiya",
        "Thaldena",
        "Horowpothana",
        "Yaka Wewa",
        "Thanthirimale",
        "Galgamuwa",
        "Moragaswewa",
        "Badalgama",
        "Giriulla",
        "Dunamale",
        "Baduruwadiya",  # harmless extras; parser ignores if unused
        "Akuressa",
        "Sapugoda",
        "Thimbolketiya",
        "Tissa Wewa",
        "Menik Farm",
        "Padawiya Tank",
        "Kalawana",
        "Daisy Valley",
        "Amunugama",
        "Ethiliyagala",
        "Ma Eliya",
        "Diniminiyathanna",
        "Hatharaliyadde",
        "Molagoda",
        "Bolagama Bridge",
        "Dambulu Oya River",
        "Bibile",
        "Kandalama",
        "Rajanganaya",
        "Buttala Bridge",
        "Usgala Siyambalangamuwa",
        "Castlereigh",
        "Norton",
        "Maussakelle",
        "Canyon",
        "Laxapana",
        "Colombo",
        "Inginiyagala",
        "Kotmale",
        "Victoria",
        "Kukulegama",
        "Randenigala",
        "Rantembe",
        "Bowatenna",
        "Ukuwela",
        "U.Kothmale",
        "Parakrama Samudra",
        "Minneriya",
        "Kaudulla",
        "Ulhitiya",
        "Aranayake",
        "Girithale",
        "Diyabeduma",
        "Ulhitiya Inlet Canal",
        "Minipe LB Canal",
        "Lankagama",
        "Deniyaya Bridge",
        "Hurulu Wewa",
        "Akuressa",
        "Sapugoda",
        "Mau Ara",
        "Nachchaduwa",
        "Samanalawewa",
        "Usgala Siyamabalangamuwa",
        "Bolgoda Lake",
        "Halwathura",
        "Anguruwathota",
        "Dela",
        "Magura (Baduraliya)",
        "Malwala",
        "Banagoda",
        "Watapotha",
        "Paragoda",
        "Aviththawa",
        "Diyabeduma",
        "Benthara Samudra",
        "Minneriya",
        "Parakrama Samu. (Res)",
        "Maha Oya",
        "Attanagalla",
        "Ampara",
    },
    key=len,
    reverse=True,
)

# Words/lines to ignore from PDF pages.
SKIP_LINE_PREFIXES = (
    "Islandwide Water Level",
    "DATE :",
    "Prepared by :",
    "Prepared by:",
    "Checked by :",
    "Checked by:",
    "Director (Hydrology and Disaster Management)",
    "Tributory/River",
    "Gauging Station",
    "River Basin",
    "Water Level at",
    "24 Hr RF",
    "Remarks",
    "Level at",
    "Minor Flood",
    "Major Flood",
    "Water Level Rising or Falling",
    "Water Level",
    "Units in mm",
    "Daily Rainfall",
    "Notation",
)

WATER_ROW_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<unit>\bft\b|\bm\b)\s+"
    r"(?P<alert>NA|-?\d+(?:\.\d+)?)\s+"
    r"(?P<minor>NA|-?\d+(?:\.\d+)?)\s+"
    r"(?P<major>NA|-?\d+(?:\.\d+)?)\s+"
    r"(?P<wl8>NA|-?\d+(?:\.\d+)?)\s+"
    r"(?P<wl9>NA|-?\d+(?:\.\d+)?)\s+"
    r"(?P<remark>[A-Za-z][A-Za-z\s\-()/]*)\s+"
    r"(?P<rainfall>NA|-?\d+(?:\.\d+)?|-)$"
)

DATE_RE = re.compile(r"DATE\s*:\s*(?P<date>\d{1,2}-[A-Za-z]{3}-\d{4})", re.I)
TIME_RE = re.compile(
    r"TIME\s*:\s*(?P<time>\d{1,2}:\d{2}(?:\s*[AP]M)?)", re.I
)
LIST_ROW_RE = re.compile(
    r"(?P<title>.+?)\s+(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>\d{1,2}:\d{2}(?:\s*[AP]M)?)\s+Download",
    re.I,
)
REPORT_DATE_RANGE_RE = re.compile(
    r"\(\s*24 hrs ending at\s+(?P<endtime>\d{1,2}:\d{2}\s*[AP]M)\s+on\s+(?P<enddate>\d{1,2}-[A-Za-z]{3}-\d{4})\s*\)",
    re.I,
)
PAIR_RE = re.compile(
    r"(?P<station>[A-Z][A-Za-z0-9()./\-,' ]*?)\s+(?P<value>NA|-?\d+(?:\.\d+)?)"
)
RB_CODE_RE = re.compile(r"\(RB\s*\d+\)", re.I)


@dataclasses.dataclass
class ReportItem:
    title: str
    date: str
    time: str
    url: str
    source_text: str


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    return session


def parse_date_any(value: str) -> Optional[dt.datetime]:
    value = value.strip()
    fmts = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M %p",
        "%d-%b-%Y %H:%M",
        "%d-%b-%Y %I:%M %p",
        "%d-%b-%Y",
        "%Y-%m-%d",
    ]
    for fmt in fmts:
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def safe_float(value: str):
    value = (value or "").strip()
    if value in {"", "-", "NA"}:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def pages_to_lines(page: pdfplumber.page.Page, y_tolerance: float = 2.2) -> list[str]:
    """
    Convert a PDF page to text lines using word coordinates.
    This is more robust than plain extract_text() for multi-column bulletins.
    """
    words = page.extract_words(
        keep_blank_chars=False,
        use_text_flow=True,
        extra_attrs=["x0", "top"],
    )
    if not words:
        return []
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_top: Optional[float] = None
    for w in words:
        top = float(w["top"])
        if current_top is None or abs(top - current_top) <= y_tolerance:
            current.append(w)
            if current_top is None:
                current_top = top
            else:
                current_top = (current_top * (len(current) - 1) + top) / len(current)
        else:
            groups.append(current)
            current = [w]
            current_top = top
    if current:
        groups.append(current)

    lines = []
    for group in groups:
        line = " ".join(w["text"] for w in group)
        line = normalize_space(line)
        if line:
            lines.append(line)
    return lines


def is_header_or_footer(line: str) -> bool:
    line_n = normalize_space(line)
    if not line_n:
        return True
    if line_n.startswith(SKIP_LINE_PREFIXES):
        return True
    if line_n.startswith("(") and "24 hrs ending" in line_n.lower():
        return True
    if line_n in {
        "NA :- Not Available",
        "RB :- River Basin",
        "MET :- Department of Meteorology",
        "ID :- Irrigation Department",
        "CEB :- Ceylon Electricity Board",
        "Res :- Reservoir",
        "HMIS :- DSWRP Data",
        "Tr :- Traced Rainfall",
    }:
        return True
    return False


def clean_prefix(prefix: str) -> str:
    prefix = normalize_space(prefix)
    prefix = RB_CODE_RE.sub("", prefix)
    prefix = normalize_space(prefix)
    return prefix


def split_station_from_prefix(prefix: str) -> tuple[str | None, str]:
    """
    Split a line prefix into (station_name, remainder_text) by matching the
    longest known station suffix.
    """
    prefix = clean_prefix(prefix)
    for station in KNOWN_STATIONS:
        if prefix.endswith(station):
            remainder = normalize_space(prefix[: -len(station)])
            return station, remainder
    return None, prefix


def parse_report_list(html: str) -> list[ReportItem]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[ReportItem] = []
    seen_urls: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/images/dmcreports/" not in href.lower():
            continue
        url = urljoin(BASE_URL, href)
        if url in seen_urls:
            continue

        parent = a.find_parent(["tr", "li", "div"]) or a.parent
        text = normalize_space(" ".join(parent.stripped_strings)) if parent else normalize_space(a.get_text(" "))
        m = LIST_ROW_RE.search(text)
        if not m:
            # Fallback: use the anchor text and any nearby text.
            # This keeps the scraper resilient if DMC tweaks the HTML slightly.
            title = normalize_space(a.get_text(" ")) or "DMC Report"
            date = ""
            time_str = ""
        else:
            title = normalize_space(m.group("title"))
            date = m.group("date")
            time_str = m.group("time")

        # Some rows are repeated or may include extra whitespace.
        seen_urls.add(url)
        items.append(
            ReportItem(
                title=title,
                date=date,
                time=time_str,
                url=url,
                source_text=text,
            )
        )

    # Deduplicate by URL while preserving order.
    unique: dict[str, ReportItem] = {}
    for item in items:
        unique.setdefault(item.url, item)
    return list(unique.values())


def download_file(session: requests.Session, url: str, dest: Path, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            with session.get(url, stream=True, timeout=90) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
            return dest
        except Exception as e:
            if i == retries - 1:
                raise e
            time.sleep(2 * (i + 1))
    return dest


def extract_pdf_metadata(page_text: str) -> dict[str, str]:
    meta = {}
    m_date = DATE_RE.search(page_text)
    if m_date:
        meta["report_date_text"] = m_date.group("date")
    m_time = TIME_RE.search(page_text)
    if m_time:
        meta["report_time_text"] = m_time.group("time")
    m_range = REPORT_DATE_RANGE_RE.search(page_text)
    if m_range:
        meta["rainfall_end_date_text"] = m_range.group("enddate")
        meta["rainfall_end_time_text"] = m_range.group("endtime")
    return meta


def parse_water_level_page(
    lines: list[str],
    report_meta: dict[str, str],
    report_url: str,
    report_title: str,
    page_number: int,
) -> list[dict]:
    """
    Parse the first page into row-level water data.
    """
    records: list[dict] = []
    current_basin: Optional[str] = None

    i = 0
    while i < len(lines):
        line = normalize_space(lines[i])

        if not line or is_header_or_footer(line):
            i += 1
            continue

        # Basin heading lines sometimes appear as a basin name alone or with RB code.
        if (
            "m " not in f" {line} "
            and " ft " not in f" {line} "
            and not re.search(r"\d", line)
            and len(line.split()) <= 5
        ):
            # Example: "Kelani Ganga" / "(RB 01)"
            if "RB" not in line and not line.startswith("("):
                current_basin = line
            i += 1
            continue

        # Try to merge with the next line if this line is clearly wrapped.
        candidate = line
        if i + 1 < len(lines):
            nxt = normalize_space(lines[i + 1])
            if nxt and not is_header_or_footer(nxt):
                # Merge if the current line does not yet include the numeric columns.
                if re.search(r"\b(?:m|ft)\b", candidate) and len(re.findall(r"(?<![A-Za-z])(?:-?\d+(?:\.\d+)?|NA)(?![A-Za-z])", candidate)) < 7:
                    candidate = normalize_space(candidate + " " + nxt)
                    i += 1

        m = WATER_ROW_RE.match(candidate)
        if not m:
            # Keep raw lines in a fallback record so data is not silently lost.
            records.append(
                {
                    "report_title": report_title,
                    "report_url": report_url,
                    "page_number": page_number,
                    "record_type": "water_level_raw",
                    "report_date_text": report_meta.get("report_date_text"),
                    "report_time_text": report_meta.get("report_time_text"),
                    "raw_text": line,
                }
            )
            i += 1
            continue

        prefix = m.group("prefix")
        station_name, prefix_remainder = split_station_from_prefix(prefix)

        # If the row starts with a known current basin, strip it from the prefix.
        basin = current_basin
        tributory_river = prefix_remainder

        if basin and tributory_river.startswith(basin):
            tributory_river = normalize_space(tributory_river[len(basin):])

        # When the prefix itself contains the basin name, infer it directly.
        if not basin:
            for basin_candidate in [
                "Kelani Ganga",
                "Kalu Ganga",
                "Gin Ganga",
                "Nilwala Ganga",
                "Walawe Ganga",
                "Kirindi Oya",
                "Menik Ganga",
                "Kumbukkan Oya",
                "Heda Oya",
                "Maduru Oya",
                "Mahaweli Ganga",
                "Yan Oya",
                "Maa Oya",
                "Malwathu Oya",
                "Mee Oya",
                "Deduru Oya",
                "Maha Oya",
                "Attanagalu Oya",
                "Benthara Ganga",
                "Bolgoda Ganga",
                "Daduru Oya",
                "Kala Oya",
            ]:
                if prefix.startswith(basin_candidate):
                    basin = basin_candidate
                    tributory_river = normalize_space(prefix[len(basin_candidate):])
                    break

        # If we only found a station name, keep the rest as a free-form location text.
        if not station_name:
            station_name = prefix_remainder or prefix

        records.append(
            {
                "report_title": report_title,
                "report_url": report_url,
                "page_number": page_number,
                "record_type": "water_level",
                "report_date_text": report_meta.get("report_date_text"),
                "report_time_text": report_meta.get("report_time_text"),
                "river_basin": basin,
                "tributory_river": tributory_river or None,
                "gauging_station": station_name,
                "unit": m.group("unit"),
                "alert_level": safe_float(m.group("alert")),
                "minor_flood_level": safe_float(m.group("minor")),
                "major_flood_level": safe_float(m.group("major")),
                "water_level_8am": safe_float(m.group("wl8")),
                "water_level_9am": safe_float(m.group("wl9")),
                "water_level_status": normalize_space(m.group("remark")),
                "rainfall_24hr_mm": safe_float(m.group("rainfall")),
                "raw_text": candidate,
            }
        )

        i += 1

    return records


def parse_rainfall_page(
    lines: list[str],
    report_meta: dict[str, str],
    report_url: str,
    report_title: str,
    page_number: int,
) -> list[dict]:
    """
    Parse the rainfall page into station/rainfall rows.
    This page is more free-form, so we use a pair-finding regex and keep raw
    lines when the structure is unclear.
    """
    records: list[dict] = []
    for line in lines:
        line = normalize_space(line)
        if not line or is_header_or_footer(line):
            continue
        if line.startswith("DATE :") or line.startswith("("):
            continue
        if "Notation" in line:
            continue
        # Find all station/value pairs on the line.
        pairs = list(PAIR_RE.finditer(line))
        if not pairs:
            records.append(
                {
                    "report_title": report_title,
                    "report_url": report_url,
                    "page_number": page_number,
                    "record_type": "rainfall_raw",
                    "report_date_text": report_meta.get("report_date_text"),
                    "report_time_text": report_meta.get("report_time_text"),
                    "rainfall_end_date_text": report_meta.get("rainfall_end_date_text"),
                    "rainfall_end_time_text": report_meta.get("rainfall_end_time_text"),
                    "raw_text": line,
                }
            )
            continue

        for m in pairs:
            station = normalize_space(m.group("station"))
            value = m.group("value")
            # Filter out obvious false positives from headings.
            if len(station) < 2:
                continue
            if station in {"Tr", "RB", "MET", "ID", "CEB", "Res", "HMIS", "NA"}:
                continue
            records.append(
                {
                    "report_title": report_title,
                    "report_url": report_url,
                    "page_number": page_number,
                    "record_type": "rainfall",
                    "report_date_text": report_meta.get("report_date_text"),
                    "report_time_text": report_meta.get("report_time_text"),
                    "rainfall_end_date_text": report_meta.get("rainfall_end_date_text"),
                    "rainfall_end_time_text": report_meta.get("rainfall_end_time_text"),
                    "station": station,
                    "rainfall_mm": safe_float(value),
                    "raw_text": line,
                }
            )

    return records


def parse_generic_pdf(
    pdf_path: Path,
    report_url: str,
    report_title: str,
) -> list[dict]:
    """
    Parse a PDF report. If the report matches the water-level bulletin structure,
    extract row-level data; otherwise, return page-level raw records.
    """
    records: list[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_texts = [page.extract_text() or "" for page in pdf.pages]
        page_lines = [pages_to_lines(page) for page in pdf.pages]

        meta = {}
        if page_texts:
            meta = extract_pdf_metadata(page_texts[0])

        # Water-level bulletin (page 0)
        if page_lines:
            lines0 = page_lines[0]
            records.extend(
                parse_water_level_page(
                    lines0, meta, report_url, report_title, page_number=1
                )
            )

        # Rainfall page (page 1)
        if len(page_lines) > 1:
            meta2 = dict(meta)
            meta2.update(extract_pdf_metadata(page_texts[1] if len(page_texts) > 1 else ""))
            records.extend(
                parse_rainfall_page(
                    page_lines[1], meta2, report_url, report_title, page_number=2
                )
            )

        # Any remaining pages: save raw text.
        for idx in range(2, len(page_lines)):
            for line in page_lines[idx]:
                if not line or is_header_or_footer(line):
                    continue
                records.append(
                    {
                        "report_title": report_title,
                        "report_url": report_url,
                        "page_number": idx + 1,
                        "record_type": "raw_page_line",
                        "report_date_text": meta.get("report_date_text"),
                        "report_time_text": meta.get("report_time_text"),
                        "raw_text": line,
                    }
                )

    return records


def parse_csv_report(
    csv_path: Path,
    report_url: str,
    report_title: str,
    report_date_text: str | None,
    report_time_text: str | None,
) -> list[dict]:
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "report_title": report_title,
                "report_url": report_url,
                "record_type": "csv_row",
                "report_date_text": report_date_text,
                "report_time_text": report_time_text,
                **{f"col_{k}": v for k, v in row.to_dict().items()},
            }
        )
    return records


def pick_output_filename(report_url: str, default_ext: str = ".pdf") -> str:
    # URL structure: .../Water_level_&_Rainfall_2025__1764294063.pdf
    # We want to preserve the ID or use a hash to ensure uniqueness.
    url_path = report_url.split("?")[0]
    base = url_path.split("/")[-1]
    
    if "." not in base:
        base += default_ext
        
    # Standardize filename but keep ID
    # Preserve the __ID part if it exists
    base = re.sub(r"[^a-zA-Z0-9._\-& ]", "_", base)
    
    # If the filename still seems generic (less than 10 chars), prefix with url hash
    if len(base) < 10:
        h = hashlib.md5(report_url.encode()).hexdigest()[:8]
        base = f"{h}_{base}"
        
    return base


def scrape_reports(
    report_type_id: int,
    output_csv: Path,
    downloads_dir: Path,
    list_url: str | None = None,
    max_reports: int | None = None,
    since: str | None = None,
    until: str | None = None,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    session = make_session()
    url = list_url or DEFAULT_LIST_URL.format(report_type_id=report_type_id)
    resp = session.get(url, timeout=60)
    resp.raise_for_status()

    items = parse_report_list(resp.text)
    if since:
        since_dt = parse_date_any(since)
        if since_dt is None:
            raise ValueError(f"Could not parse --from-date value: {since}")
        items = [
            x for x in items
            if parse_date_any(x.date) and parse_date_any(x.date) >= since_dt
        ]
    if until:
        until_dt = parse_date_any(until)
        if until_dt is None:
            raise ValueError(f"Could not parse --to-date value: {until}")
        items = [
            x for x in items
            if parse_date_any(x.date) and parse_date_any(x.date) <= until_dt
        ]

    if max_reports is not None:
        items = items[:max_reports]

    all_records: list[dict] = []

    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] {item.date} {item.time} | {item.title}")
        ext = Path(item.url.split("?")[0]).suffix.lower() or ".pdf"
        filename = pick_output_filename(item.url, default_ext=ext)
        local_path = downloads_dir / filename

        try:
            if not local_path.exists():
                download_file(session, item.url, local_path)
            else:
                print(f"  - using cached file: {local_path.name}")

            if ext == ".csv":
                records = parse_csv_report(
                    local_path,
                    item.url,
                    item.title,
                    report_date_text=item.date,
                    report_time_text=item.time,
                )
            else:
                records = parse_generic_pdf(local_path, item.url, item.title)

            # Attach list-page metadata to all records.
            for rec in records:
                rec.setdefault("list_title", item.title)
                rec.setdefault("list_date", item.date)
                rec.setdefault("list_time", item.time)
                rec.setdefault("source_file", local_path.name)
            all_records.extend(records)

        except Exception as e:
            all_records.append(
                {
                    "list_title": item.title,
                    "list_date": item.date,
                    "list_time": item.time,
                    "report_title": item.title,
                    "report_url": item.url,
                    "source_file": local_path.name,
                    "record_type": "error",
                    "error_message": str(e),
                }
            )

        time.sleep(sleep_seconds)

    df = pd.DataFrame(all_records)

    # Normalize columns for a consistent CSV.
    preferred_order = [
        "list_date",
        "list_time",
        "list_title",
        "report_title",
        "report_url",
        "source_file",
        "record_type",
        "page_number",
        "report_date_text",
        "report_time_text",
        "rainfall_end_date_text",
        "rainfall_end_time_text",
        "river_basin",
        "tributory_river",
        "gauging_station",
        "station",
        "unit",
        "alert_level",
        "minor_flood_level",
        "major_flood_level",
        "water_level_8am",
        "water_level_9am",
        "water_level_status",
        "rainfall_24hr_mm",
        "rainfall_mm",
        "raw_text",
        "error_message",
    ]
    cols = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    df = df[cols] if not df.empty else pd.DataFrame(columns=preferred_order)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape DMC reports into a CSV file.")
    parser.add_argument(
        "--report-type-id",
        type=int,
        default=6,
        help="DMC report_type_id. Default: 6 (River Water Level and Flood Warning).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dmc_reports.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path("dmc_downloads"),
        help="Directory to cache downloaded reports.",
    )
    parser.add_argument(
        "--list-url",
        type=str,
        default=None,
        help="Override the DMC report list URL.",
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Keep only list-page entries on/after this date (e.g. 2025-12-01).",
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="Keep only list-page entries on/before this date (e.g. 2026-03-23).",
    )
    parser.add_argument(
        "--max-reports",
        type=int,
        default=None,
        help="Process only the first N reports from the filtered list.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between downloads.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.downloads_dir.mkdir(parents=True, exist_ok=True)

    df = scrape_reports(
        report_type_id=args.report_type_id,
        output_csv=args.output,
        downloads_dir=args.downloads_dir,
        list_url=args.list_url,
        max_reports=args.max_reports,
        since=args.from_date,
        until=args.to_date,
        sleep_seconds=args.sleep,
    )

    print(f"\nSaved {len(df)} rows to: {args.output.resolve()}")
    if not df.empty:
        print("Columns:", ", ".join(df.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
