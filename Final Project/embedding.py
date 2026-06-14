import gc, time, faiss, numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def build_index(chunks: list, index_path: str):
    start = time.time()
    print("[Embedding] Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    print("[Embedding] Encoding chunks...")
    embeddings = model.encode(chunks, batch_size=8, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)
    print(f"[Embedding] Index saved to {index_path} in {time.time()-start:.1f}s")
    del model, embeddings
    gc.collect()
    return index

if __name__ == "__main__":
    from ingestion import ingest
    chunks = ingest("/home/husey/edgerag/data")
    build_index(chunks, "/home/husey/edgerag/index/faiss.index")
