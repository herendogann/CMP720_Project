# EdgeRAG: Offline Retrieval-Augmented Generation on Resource-Constrained Simulated ARM Architectures 🚀

**Institution:** Hacettepe University -  Computer Engineering  
**Course:** CMP720 - Embedded System Design - Spring 2026  
**Author:** Hüseyin Eren Doğan

---

## 📌 Project Overview
Deploying Large Language Models (LLMs) usually requires high-bandwidth cloud infrastructure. However, embedded environments (industrial IoT, remote sensors, air-gapped facilities) demand systems that can reason and answer queries offline due to connectivity, privacy, and security constraints. 

**EdgeRAG** aims to solve this by building a fully offline Retrieval-Augmented Generation (RAG) pipeline designed specifically for resource-constrained edge environments.

## ⚙️ System Architecture & Constraints
Instead of relying on physical hardware, this project utilizes a strict simulation environment to test the theoretical and practical limits of the software stack:
* **Hardware Emulator:** QEMU simulating an ARM-based Single Board Computer (SBC).
* **Resource Constraints:** Artificially limited RAM (e.g., 2GB - 4GB) and restricted CPU processing power.
* **Vector Database:** Lightweight local instances (FAISS / ChromaDB).
* **Language Model:** Heavily quantized Small Language Model (SLM) in GGUF format (e.g., 4-bit Phi-3 or TinyLlama).
* **Orchestration:** Minimal LangChain architecture for document chunking, embedding, and retrieval.

## 🛤️ Project Milestones (9-10 Weeks Timeline)
The project will be developed and evaluated through the following core phases:

1. **Phase 1: Environment & Emulation Setup**
2. **Phase 2: Local Model Deployment**
3. **Phase 3: Offline RAG Integration**
4. **Phase 4: Profiling & System Evaluation**
5. **Phase 5: Finalization & Reporting**
