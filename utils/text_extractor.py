import os
import logging
import PyPDF2
import docx2txt

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path):
    """Extract text from PDF with enhanced error handling"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                try:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + " "
                except Exception as page_error:
                    logger.warning(f"Error extracting page {page_num} from PDF: {str(page_error)}")
                    continue
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text from {file_path}: {str(e)}")
        return ""


def extract_text_from_docx(file_path):
    """Extract text from DOCX with enhanced error handling"""
    try:
        text = docx2txt.process(file_path)
        return text.strip() if text else ""
    except Exception as e:
        logger.error(f"Error extracting DOCX text from {file_path}: {str(e)}")
        return ""


def extract_text_from_txt(file_path):
    """Extract text from TXT with multiple encoding support"""
    try:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read().strip()
            except UnicodeDecodeError:
                continue
        logger.warning(f"Could not decode text file: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting TXT text from {file_path}: {str(e)}")
        return ""


def extract_text(file_path):
    """Enhanced text extraction with comprehensive error handling"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            return extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return extract_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return ""

    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""