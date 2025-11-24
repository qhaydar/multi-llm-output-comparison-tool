# multi-llm-output-comparison-tool

[![Python Version](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)

A Python-based utility for comparing outputs from multiple Large Language Models (LLMs) given the same prompt.  
Use it to test, evaluate, and contrast how different LLMs respond to identical input.

---

## 🚀 Features

- Send a prompt once and receive responses from multiple LLMs
- Compare outputs side‑by‑side for easy inspection
- Track metadata like tokens used, **latency (time taken for each model)**, and cost (if supported)
- **Each model’s response now includes the elapsed time in seconds**
- Useful for:
  - Model selection
  - Prompt engineering
  - Quality or consistency checking across LLM providers

---

## 🛠 Getting Started

### Requirements

- Python 3.8 or newer  
- API credentials for your LLM providers (e.g., OpenAI, Anthropic, Mistral)  

Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

1. Create an environment file (e.g., `.env`) with your API keys:
```bash
OPENAI_API_KEY=<your_openai_key>
# Add other provider keys as needed
```

2. Adjust provider/model settings in `main.py` or configuration files.

### Usage

Run the tool from the command line:

```bash
python main.py --prompt "Your prompt here" --models openai:gpt-4,anthropic:claude-2
```

Example:

```bash
python main.py --prompt "Summarize the lifecycle of a butterfly." --models openai:gpt-4,anthropic:claude-2,mistral:mixtral-8x7b
```

Results are printed or saved to files for comparison.

---

## ✅ Typical Workflow

1. Define prompt(s)  
2. Select LLMs to compare  
3. Run the tool and collect results  
4. Inspect side-by-side:
   - Which model gave the most relevant answer?  
   - Which style is better?  
   - Compare speed, tokens, and cost  
5. Iterate on prompts or models

---

## 📂 Project Structure

```
multi-llm-output-comparison-tool/
├── main.py            ← Entry point for running comparisons
├── providers/         ← Modules to integrate LLM APIs
├── config/            ← Configuration files
├── results/           ← Output directory for comparison results
└── README.md
```

---

## 🔍 Why Use This Tool?

- Compare multiple LLMs simultaneously  
- Improve prompt-engineering with cross-model testing  
- Ensure quality and consistency across providers  
- Evaluate cost/performance tradeoffs  

---

## ✍️ Contributing

Contributions welcome! To contribute:

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit changes (`git commit -m "Add feature XYZ"`)  
4. Push (`git push origin feature-name`)  
5. Open a Pull Request describing your changes  

---

## 🧾 License

MIT License. See the `LICENSE` file.

---

## 👤 Author

Created by Q.  
Reach out via GitHub issues or pull requests.
