# EdgeRAG

**Offline Retrieval-Augmented Generation on Resource-Constrained Simulated ARM Architectures**

> CMP720 Embedded System Design — Spring 2026  
> Hacettepe University · Aselsan  
> Hüseyin Eren Doğan · N25123065

---

Presentation Video: https://youtu.be/U-SuaEX9GkQ

---

## Overview

EdgeRAG is a fully offline RAG pipeline designed for air-gapped, privacy-critical, and energy-constrained embedded environments. It runs on a QEMU-emulated ARM Cortex-A53 (AArch64) with a hard 2 GB RAM ceiling and zero network access.

The pipeline combines 4-bit quantized LLM inference (Phi-3 Mini via llama.cpp), CPU-optimized vector retrieval (FAISS), semantic chunking, and a KPN module architecture that enforces strict memory isolation between stages.

**No cloud. No GPU. No internet.**

---

## Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Query completion | ≥ 95% | **100%** (5/5) |
| Avg. end-to-end latency | ≤ 60 s | **58.6 s** |
| Token generation speed | ≥ 0.5 t/s | **2.92 t/s** |
| Peak RAM usage | ≤ 1800 MB | **729 MB** |
| OOM failure rate | < 5% | **0%** |



---

## Tech Stack

| Component | Choice |
|-----------|--------|
| LLM | Phi-3 Mini Q4_K_M (GGUF) |
| Inference | llama-cpp-python (CPU-only) |
| Vector DB | FAISS flat L2 |
| Embeddings | all-MiniLM-L6-v2 |
| Orchestration | LangChain |
| Platform | QEMU ARM Cortex-A53 (AArch64) |

---

## License

Academic project — Hacettepe University CMP720, Spring 2026.
