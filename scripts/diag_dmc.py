import requests
from bs4 import BeautifulSoup
import re

BASE_URL = "https://www.dmc.gov.lk"
LIST_URL = BASE_URL + "/index.php?Itemid=277&lang=en&option=com_dmcreports&report_type_id=6&view=reports&limit=0"

def get_unique_urls():
    print(f"Fetching list from {LIST_URL}...")
    resp = requests.get(LIST_URL, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/images/dmcreports/" in href.lower():
            urls.append(href)
            
    print(f"Total report links found: {len(urls)}")
    print(f"Unique report URLs: {len(set(urls))}")
    
    # Check first few URLs
    for u in list(set(urls))[:10]:
        print(f"Example URL: {u}")

if __name__ == "__main__":
    get_unique_urls()
