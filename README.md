# HealthWave: AI-Powered Medical Document Simplification Platform

## Overview

HealthWave bridges the gap between complex medical language and patient understanding. Our platform leverages advanced AI models to translate technical medical terminology into clear, accessible language, empowering patients and caregivers with better health literacy.

## Core Features

### 1. Medical Text Translator
Converts complex medical terminology into plain language explanations.

### 2. AI Medical Assistant
An intelligent chatbot that answers questions about medical terms and conditions using a comprehensive knowledge base.

### 3. Medical Report Analyzer
Extracts and summarizes information from medical reports (images or PDFs) into easy-to-understand summaries.

## Quick Start Guide

### Prerequisites

- **Ollama** with Llama3.2 model installed and running on port 11434

  Install Ollama ([Download here](https://ollama.com/download)):
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

  Download Llama3 model:
  ```bash
  ollama install llama3
  ```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/losthumanity/healthWave.git
   cd healthWave
   ```

2. **Setup Database**
   ```bash
   cd database
   docker compose up -d
   ```

3. **Setup Backend**
   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

4. **Run Backend**
   ```bash
   uvicorn main:medicalsearch --host 0.0.0.0 --port 8000 --reload
   ```

5. **Setup Frontend**
   ```bash
   cd frontend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

6. **Run Frontend**
   ```bash
   python3 main.py
   ```

## Technology Stack

### Medical Text Translator

Built on the **T5 (Text-To-Text Transfer Transformer)** model, fine-tuned specifically for medical terminology simplification.

#### Training Approach

- **Task Prefix**: `"simplify: "` prepended to all medical texts
- **Architecture**: T5-base model fine-tuned on curated medical text pairs

#### Data Sources

- cbasu/Med-EASi
- MTSamples
- SimMedLexSp
- PLABA
- WWW2019 medical research datasets
- LLM-augmented data generation for enhanced coverage

#### Training Configuration

- Learning rate: 3e-05
- Batch size: 4 (train/eval)
- Optimizer: Adam (β=(0.9, 0.999), ε=1e-08)
- Scheduler: Linear with 1000 warmup steps
- Epochs: 10
- Weight decay: 0.01

#### Performance Metrics

- Cross-Entropy Loss: 0.0221
- ROUGE-1: 0.8485
- ROUGE-2: 0.7157
- ROUGE-L: 0.8451

#### Quality Assurance

All model outputs are stored in the database for expert validation by healthcare professionals, with approved corrections used for continuous model improvement.

### Medical Report Analyzer

- **OCR Engine**: Pytesseract for text extraction from images and PDFs
- **Language Model**: Llama3 for intelligent text simplification
- **Output**: Concise, patient-friendly summaries

### AI Medical Assistant

- **Framework**: LangChain with Retrieval Augmented Generation (RAG)
- **Language Model**: Llama3 (via Ollama)
- **Vector Database**: FAISS-GPU for fast semantic search
- **Embeddings**: Hugging Face models for semantic understanding
- **Memory**: Session-based context management for coherent conversations

## Privacy & Security

HealthWave prioritizes user privacy with a privacy-first architecture:

- **On-Device Processing**: Sensitive information is processed locally before any cloud transmission
- **Data Anonymization**: Personal identifiers are stripped before data leaves your device
- **No Tracking**: Your medical information remains private and is never used for profiling
- **Future-Ready**: Designed with mobile-first architecture for enhanced local security

---

**License**: See [LICENSE](LICENSE) file for details.
