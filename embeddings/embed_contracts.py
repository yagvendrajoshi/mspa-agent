from langchain.text_splitter import RecursiveCharacterTextSplitter  # splits text into chunks
from sentence_transformers import SentenceTransformer  # creates embeddings
from pathlib import Path  # file paths

# load embedding model
MODEL_NAME = "BAAI/bge-large-en-v1.5"
model = SentenceTransformer(MODEL_NAME)

# text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

# read text file
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# chunk and embed
def process_contract(file_path):
    print(f"Processing: {file_path}")
    text = read_text_file(file_path)
    chunks = splitter.split_text(text)
    print(f"Total chunks: {len(chunks)}")
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    print(f"Embedding shape: {embeddings.shape}")
    return chunks, embeddings

# run on sample contract
if __name__ == "__main__":
    file_path = Path("data/sample_contracts.txt")
    chunks, embeddings = process_contract(file_path)
    
    print(f"\nFirst chunk:")
    print(chunks[0])
    print(f"\nFirst embedding (first 5 values):")
    print(embeddings[0][:5])