import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from firebase_admin import credentials, firestore, initialize_app
from utils import parse_and_chunk, embed_chunks
import flask_cors

load_dotenv()

app = Flask(__name__)
flask_cors.CORS(app)    

# Firebase initialization
cred = credentials.Certificate('fire.json')
initialize_app(cred)
db = firestore.client()

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filename = file.filename
    file_ext = filename.split('.')[-1].lower()

    if file_ext not in ['csv', 'txt', 'pdf', 'docx', 'json']:  # Add json if you handle it
        return jsonify({"error": "Unsupported file type"}), 400

    # ✅ Save file locally
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    print(f"File saved to {save_path}")

    # ✅ Extract, chunk
    chunks = parse_and_chunk(save_path, file_ext)
    if not chunks:
        return jsonify({"error": "No content extracted from file."}), 400

    # ✅ Embed
    embeddings = embed_chunks(chunks)
    print(f"Generated {len(embeddings)} embeddings")

    # ✅ Store in Firestore
    collection_ref = db.collection('document_embeddings')
    batch = db.batch()

    for idx, (chunk, embed_vector) in enumerate(zip(chunks, embeddings)):
        doc_ref = collection_ref.document()  # Auto-generated ID
        batch.set(doc_ref, {
            'filename': filename,
            'chunk_index': idx,
            'content': chunk,
            'embedding': embed_vector.tolist(),  # Convert tensor to list
        })

    batch.commit()
    print("Stored embeddings in Firestore")

    return jsonify({"message": "File processed, embedded, and stored successfully."}), 200


if __name__ == '__main__':
    app.run(debug=True)
