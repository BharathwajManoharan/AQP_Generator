import logging
import os
from typing import Optional

import pdfplumber
import docx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    try:
        text = ''
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        logger.info(f"Text extracted successfully from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        return None

def convert_pdf_to_word(pdf_path: str) -> Optional[str]:
    try:
        word_path = os.path.splitext(pdf_path)[0] + '.docx'
        doc = docx.Document()
        text = extract_text_from_pdf(pdf_path)
        if text is None:
            raise ValueError("Text extraction from PDF failed.")
        doc.add_paragraph(text)
        doc.save(word_path)
        logger.info(f"PDF converted to Word document at {word_path}")
        return word_path
    except Exception as e:
        logger.error(f"Failed to convert PDF to Word for {pdf_path}: {e}")
        return None

def process_pdf_for_nlp(input_path: str) -> Optional[str]:
    text = extract_text_from_pdf(input_path)
    if text is None:
        logger.error("PDF processing halted due to failure in text extraction.")
        return None

    processed_data = tokenize_and_parse(text)
    output_path = os.path.splitext(input_path)[0] + '.xml'
    if save_to_xml(processed_data, output_path):
        return output_path
    else:
        return None

def tokenize_and_parse(text: str) -> dict:
    tokens = text.split()
    return {'tokens': tokens}

def save_to_xml(data: dict, output_path: str) -> bool:
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('<data>\n')
            for token in data['tokens']:
                f.write(f'  <token>{token}</token>\n')
            f.write('</data>\n')
        logger.info(f"Processed data saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save data to XML {output_path}: {e}")
        return False

if __name__ == "__main__":
    input_pdf_path = "path/to/your/document.pdf"
    processed_output_path = process_pdf_for_nlp(input_pdf_path)
    if processed_output_path:
        logger.info(f"PDF processed successfully. Output saved to {processed_output_path}")
    else:
        logger.info("PDF processing was unsuccessful.")
