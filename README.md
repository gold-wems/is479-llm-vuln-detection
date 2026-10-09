# LLM-Based Security Vulnerability Detection with RAG

IS 479 Independent Study, UMBC, Fall 2026
Student: Gold Wems · Advisor: Dr. Sreedevi Sampath

## About

This project studies how well large language models (LLMs) detect security vulnerabilities in C/C++ code, and whether retrieval-augmented generation (RAG) prompting improves detection. We evaluate models from three families (ChatGPT, Gemini and Claude) on the PrimeVul paired test set, report precision, recall, F1 and pair-wise accuracy, and analyze the cases where the models fail.

**Current status:** zero-shot baseline script (`detect_vuln.py`), tested in dry-run mode. The RAG version is in design.

## Repository contents

| Path | What it is |
|---|---|
| `detect_vuln.py` | Sends PrimeVul functions to an LLM, reads its answer, and computes precision, recall, F1 and pair-wise accuracy. Every prompt and response is logged. |
| `data/primevul_test_paired.jsonl` | PrimeVul v0.1 paired test set: 870 C/C++ functions (435 vulnerable, 435 patched) |
| `data/README.md` | Where the data comes from, and its license and citation |
| `requirements.txt` | Python packages needed |

## How to run

```bash
pip install -r requirements.txt

# Test the whole pipeline without calling a model (free)
python detect_vuln.py --data data/primevul_test_paired.jsonl --limit 10 --dry-run

# Real run: needs an OpenAI API key in the OPENAI_API_KEY environment variable
python detect_vuln.py --data data/primevul_test_paired.jsonl --limit 10
```

API keys are read from environment variables and are never stored in this repository.

## Dataset

PrimeVul: Ding et al., "Vulnerability Detection with Code Language Models: How Far Are We?", ICSE 2025. https://github.com/DLVulDet/PrimeVul (MIT license). See `data/README.md`.
