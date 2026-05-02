import docx
import sys

def extract_text(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

if __name__ == '__main__':
    file_path = sys.argv[1]
    out_path = sys.argv[2]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(extract_text(file_path))
