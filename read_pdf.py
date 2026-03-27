import sys
import subprocess
try:
    import PyPDF2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf2"])
    import PyPDF2

def read_pdf(file_path):
    try:
        reader = PyPDF2.PdfReader(file_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return "\n".join(text)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    content = read_pdf("COS30019-2025 S1-A2_B.pdf")
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("Done")
