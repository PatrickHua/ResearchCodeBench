import base64
import io
import fitz

def encode_pdf_to_base64(pdf_path):
    """Encode a PDF file to a base64 string."""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_binary = pdf_file.read()
    
    # Encode to base64
    pdf_base64 = base64.b64encode(pdf_binary).decode('utf-8')
    return pdf_base64

def decode_base64_to_pdf(base64_string, output_path=None):
    """Decode a base64 string back to a PDF file."""
    # Decode from base64
    pdf_binary = base64.b64decode(base64_string)
    
    # Write to file

    
    if output_path is not None:
        with open(output_path, 'wb') as pdf_file:
            pdf_file.write(pdf_binary)
        return output_path
    else:
        return pdf_binary


def load_pdf_from_binary(pdf_binary):
    # Create a file-like object from the binary data
    memory_stream = io.BytesIO(pdf_binary)
    
    # Open the PDF from the memory stream
    pdf_document = fitz.open(stream=memory_stream, filetype="pdf")
    
    # Now you can work with the PDF
    text = ""
    for page_num in range(len(pdf_document)):
        text += pdf_document[page_num].get_text()
    
    return pdf_document, text

def decode_base64_to_pdf_and_text(base64_string):
    pdf_binary = decode_base64_to_pdf(base64_string, output_path=None)
    pdf_document, text = load_pdf_from_binary(pdf_binary)
    return pdf_document, text