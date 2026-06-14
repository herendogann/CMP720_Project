import faiss, numpy as np, gc, time
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def retrieve(query: str, chunks: list, index_path: str, k=3):
    start = time.time()
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    index = faiss.read_index(index_path)
    distances, indices = index.search(q_emb, k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    del model, index
    gc.collect()
    print(f"[Retrieval] Done in {time.time()-start:.1f}s")
    return results

if __name__ == "__main__":
    from ingestion import ingest
    chunks = ingest("/home/husey/edgerag/data")
    results = retrieve("What is ARM Cortex-A53?", chunks, "/home/husey/edgerag/index/faiss.index")
    for i, r in enumerate(results):
        print(f"\n[Chunk {i+1}]: {r[:200]}")
