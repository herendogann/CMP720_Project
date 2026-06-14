import sys, time, psutil, gc, pickle, os
sys.path.insert(0, '/home/husey/edgerag/pipeline')
from ingestion import ingest
from embedding import build_index
from retrieval import retrieve
from generation import generate

INDEX_PATH = "/home/husey/edgerag/index/faiss.index"
DATA_DIR = "/home/husey/edgerag/data"
CHUNKS_FILE = "/tmp/chunks.pkl"

TEST_QUERIES = [
    "What is an interrupt handler?",
    "How does memory protection work?",
    "What is the difference between ARM Thumb and ARM instructions?",
    "Explain the pipeline stages of a processor.",
    "What is DVFS in embedded systems?"
]

def measure_ram():
    return psutil.Process().memory_info().rss / 1024**2

print("="*60)
print("EdgeRAG Evaluation")
print("="*60)

# Ingestion
print("\n[1/4] Ingestion...")
t0 = time.time()
if os.path.exists(CHUNKS_FILE):
    chunks = pickle.load(open(CHUNKS_FILE,'rb'))
    print(f"Loaded {len(chunks)} chunks from cache")
else:
    chunks = ingest(DATA_DIR)
    pickle.dump(chunks, open(CHUNKS_FILE,'wb'))
ingest_time = time.time() - t0
print(f"Ingestion: {ingest_time:.1f}s, {len(chunks)} chunks")

# Embedding
print("\n[2/4] Building index...")
t0 = time.time()
build_index(chunks, INDEX_PATH)
embed_time = time.time() - t0
ram_after_embed = measure_ram()
print(f"Embedding: {embed_time:.1f}s | RAM: {ram_after_embed:.0f}MB")

# Retrieval + Generation loop
print("\n[3/4] Running queries...")
latencies, speeds, ram_peaks = [], [], []
oom_count = 0

for i, query in enumerate(TEST_QUERIES):
    print(f"\nQuery {i+1}: {query}")
    t0 = time.time()
    try:
        context = retrieve(query, chunks, INDEX_PATH)
        answer, tps = generate(query, context)
        latency = time.time() - t0
        ram = measure_ram()
        latencies.append(latency)
        speeds.append(tps)
        ram_peaks.append(ram)
        print(f"Answer: {answer[:100]}...")
        print(f"Latency: {latency:.1f}s | Speed: {tps:.2f} t/s | RAM: {ram:.0f}MB")
    except MemoryError:
        oom_count += 1
        print("OOM!")
    gc.collect()

# Results
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Queries completed: {len(latencies)}/{len(TEST_QUERIES)}")
print(f"OOM failures: {oom_count}")
print(f"Avg latency: {sum(latencies)/len(latencies):.1f}s")
print(f"Avg speed: {sum(speeds)/len(speeds):.2f} t/s")
print(f"Peak RAM: {max(ram_peaks):.0f}MB")
print(f"OOM rate: {oom_count/len(TEST_QUERIES)*100:.1f}%")
print("="*60)
