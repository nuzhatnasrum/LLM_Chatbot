import os
import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text() or ""  # Avoid NoneType error
            text += page_text + "\n"
    return text.strip()

def save_extracted_text(pdf_path, output_dir="data/extracted_text/"):
    """Saves extracted text from a PDF"""
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    filename = os.path.basename(pdf_path).replace(".pdf", ".txt")
    text = extract_text_from_pdf(pdf_path)

    if text:  
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"✅ Extracted text saved at: {output_path}")
    else:
        print(f"⚠️ No text extracted from {pdf_path}")

    return filename  # ✅ Fixed return statement
save_extracted_text("data/textbook_english.pdf")
save_extracted_text("data/teachers_guide_english.pdf")
save_extracted_text("data/textbook_bangla.pdf")
save_extracted_text("data/teachers_guide_bangla.pdf")
