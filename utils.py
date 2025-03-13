import pandas as pd
import docx
import json
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text_from_file(file_path, file_type):
    """
    Extracts text from different file types: PDF, DOCX, TXT, JSON, CSV (from file path).
    """
    text = ""

    if file_type == "pdf":
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    elif file_type == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])

    elif file_type == "txt":
        with open(file_path, "r", encoding='utf-8') as file:
            text = file.read()

    elif file_type == "json":
        with open(file_path, "r", encoding='utf-8') as file:
            try:
                data = json.load(file)
                if isinstance(data, (dict, list)):
                    text = json.dumps(data, indent=2)
                elif isinstance(data, str):
                    text = data
                else:
                    text = str(data)
            except json.JSONDecodeError:
                file.seek(0)
                text = file.read().decode('utf-8')

    elif file_type == "csv":
        try:
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
        except Exception as e:
            text = f"Error processing CSV: {str(e)}"

    return text


def parse_and_chunk(file_path, file_ext, chunk_size=500):
    """
    Extract and chunk text for embedding.
    """
    text = extract_text_from_file(file_path, file_ext)
    chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


def embed_chunks(chunks):
    """
    Generate embeddings from text chunks.
    """
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings
