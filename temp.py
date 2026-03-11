import os
from pdf2image import convert_from_path

# -------- CONFIG --------

PDF_PATH = "symbols.pdf"   # your input PDF
OUT_DIR = "symbols"
DPI = 300                  # high quality for detection

# Page ranges (1-based, inclusive)
LOGO_PAGE = 1
FAKE_RANGE = range(2, 17)
REAL_RANGE = range(17, 32)

# ------------------------

def ensure_dirs():
    os.makedirs(f"{OUT_DIR}/logo", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/fake_symbol", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/real_symbol", exist_ok=True)

def convert_pdf():
    ensure_dirs()

    pages = convert_from_path(PDF_PATH, dpi=DPI)

    for i, page in enumerate(pages, start=1):
        page = page.convert("RGB")

        if i == LOGO_PAGE:
            out_path = f"{OUT_DIR}/logo/logo_page_{i}.png"

        elif i in FAKE_RANGE:
            out_path = f"{OUT_DIR}/fake_symbol/fake_page_{i}.png"

        elif i in REAL_RANGE:
            out_path = f"{OUT_DIR}/real_symbol/real_page_{i}.png"

        else:
            # ignore extra pages if any
            continue

        page.save(out_path)
        print(f"Saved: {out_path}")

    print("\nPDF conversion complete.")

if __name__ == "__main__":
    convert_pdf()
