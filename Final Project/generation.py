import gc, time, psutil, os
from llama_cpp import Llama

MODEL_PATH = "/home/husey/edgerag/models/Phi-3-mini-4k-instruct-q4.gguf"

def generate(query: str, context_chunks: list, n_ctx=512, max_tokens=150):
    start = time.time()
    context = "\n\n".join(context_chunks[:2])[:600]
    prompt = f"<|user|>\nContext: {context}\n\nQuestion: {query}<|end|>\n<|assistant|>\n"
    n_threads = os.cpu_count() or 4
    print(f"[Generation] Using {n_threads} threads...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
    out = llm(prompt, max_tokens=max_tokens, stop=["<|end|>", "<|user|>"])
    answer = out["choices"][0]["text"].strip()
    tokens = out["usage"]["completion_tokens"]
    elapsed = time.time() - start
    ram = psutil.Process().memory_info().rss / 1024**2
    print(f"[Generation] {tokens} tokens in {elapsed:.1f}s ({tokens/elapsed:.2f} t/s) | RAM: {ram:.0f}MB")
    del llm
    gc.collect()
    return answer, tokens/elapsed

if __name__ == "__main__":
    answer, tps = generate("What is an embedded system?", ["An embedded system is a dedicated computer system."])
    print(f"\nAnswer: {answer}\nSpeed: {tps:.2f} t/s")
