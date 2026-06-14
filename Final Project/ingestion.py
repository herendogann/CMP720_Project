import os, gc, time
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest(data_dir: str, chunk_size=256, chunk_overlap=32):
    start = time.time()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".pdf"):
            path = os.path.join(data_dir, fname)
            reader = PdfReader(path)
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            chunks.extend(splitter.split_text(text))
            print(f"[Ingestion] {fname}: {len(chunks)} chunks so far")
    gc.collect()
    print(f"[Ingestion] Done in {time.time()-start:.1f}s — {len(chunks)} total chunks")
    return chunks

if __name__ == "__main__":
    chunks = ingest("/home/husey/edgerag/data")
    print(chunks[0][:200] if chunks else "No chunks")
