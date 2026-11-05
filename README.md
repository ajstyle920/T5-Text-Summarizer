# 📝 T5 Text Summarizer

A lightweight web app that turns long text and conversations into short, readable summaries. It uses a fine-tuned **T5 (Text-to-Text Transfer Transformer)** model from Hugging Face, served through a **FastAPI** backend with a clean, minimal web interface.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Configuration](#%EF%B8%8F-configuration)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project shows the full path from a trained NLP model to a usable product. A T5 model fine-tuned for **abstractive dialogue summarization** is loaded once at startup and exposed through a REST API. A browser-based UI lets anyone paste text and get a summary in seconds, with no ML knowledge needed.

**Example**

> **Input:**
> Amanda: I baked cookies. Do you want some?
> Jerry: Sure!
> Amanda: I'll bring you some tomorrow :-)
>
> **Output:**
> amanda baked cookies and will bring jerry some tomorrow.

---

## ✨ Features

- 🤖 **Abstractive summarization** with a fine-tuned T5 model. It writes new sentences instead of only extracting existing ones.
- ⚡ **FastAPI backend** with automatic interactive API docs (Swagger UI and ReDoc).
- 🖥️ **Simple web UI** for pasting text and viewing the summary instantly.
- 🧹 **Built-in text cleaning** that normalizes whitespace, strips HTML tags and lowercases the input.
- 🔎 **Beam search decoding** (4 beams) for more fluent, coherent summaries.
- 🚀 **Automatic hardware acceleration** that uses Apple Silicon (MPS), NVIDIA GPU (CUDA) or CPU, whichever is available.
- ✅ **Request validation** with Pydantic schemas.

---

## 🛠 Tech Stack

| Layer          | Technology                                  |
| -------------- | ------------------------------------------- |
| Model          | T5 (`T5ForConditionalGeneration`)           |
| ML Framework   | PyTorch, Hugging Face Transformers          |
| Tokenization   | SentencePiece (`T5Tokenizer`)               |
| Backend        | FastAPI, Uvicorn                            |
| Validation     | Pydantic                                    |
| Templating     | Jinja2                                      |
| Frontend       | HTML5, CSS3, Vanilla JavaScript (Fetch API) |

---

## 📁 Project Structure

```
.
├── main.py                    # FastAPI app: model loading, preprocessing, inference, routes
├── index.html                 # Frontend UI (served by FastAPI via Jinja2)
├── requirements.txt           # Python dependencies
└── saved_T5_summary_model/    # Fine-tuned T5 model + tokenizer (see "Model Setup")
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors      # or pytorch_model.bin
    ├── spiece.model
    ├── special_tokens_map.json
    └── tokenizer_config.json
```

---

## ⚙️ How It Works

```
┌────────────┐   POST /summarize    ┌──────────────────────────────────────────┐
│  Browser   │ ───────────────────▶ │                FastAPI                   │
│ (index.html)│   {"dialogue": ...}  │                                          │
│            │                      │  1. clean_data()   → normalize text      │
│            │                      │  2. T5Tokenizer    → token IDs (≤512)    │
│            │                      │  3. model.generate → beam search (4)     │
│            │ ◀─────────────────── │  4. tokenizer.decode → summary text      │
└────────────┘  {"summary": ...}    └──────────────────────────────────────────┘
```

1. **Preprocessing:** the input is cleaned (line breaks and extra spaces collapsed, HTML tags removed, text lowercased) so it matches the format the model was trained on.
2. **Tokenization:** the cleaned text is tokenized and truncated to **512 tokens**.
3. **Generation:** the model produces a summary of up to **150 tokens** using **beam search** with early stopping.
4. **Decoding:** token IDs are converted back to text and returned as JSON.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+** (developed on 3.12)
- `pip`
- *(Optional)* An NVIDIA GPU with CUDA, or an Apple Silicon Mac, for faster inference

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 For GPU support, install the PyTorch build that matches your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Model Setup

The app loads the model from a local folder named **`saved_T5_summary_model/`** in the project root. Model weights are large and usually **not committed to Git**, so set this up in one of these ways:

**Option A: use your own fine-tuned model**

Put your saved model in the project root. If you trained it yourself, save it like this:

```python
model.save_pretrained("./saved_T5_summary_model")
tokenizer.save_pretrained("./saved_T5_summary_model")
```

**Option B: download from the Hugging Face Hub**

If the model is hosted on the Hub, replace the local path in `main.py`:

```python
model = T5ForConditionalGeneration.from_pretrained("<hf-username>/<model-name>")
tokenizer = T5Tokenizer.from_pretrained("<hf-username>/<model-name>")
```

**Option C: try it quickly with a base model**

For a quick test without fine-tuning, use `"t5-small"` or `"t5-base"`. Summary quality will be lower than with the fine-tuned model.

### 5. Run the server

```bash
uvicorn main:app --reload
```

The app will be running at **http://127.0.0.1:8000** 🎉

---

## 💻 Usage

### Web Interface

1. Open **http://127.0.0.1:8000** in your browser.
2. Paste or type your text or conversation into the text box.
3. Click **Summarize**.
4. The summary appears under **Content Summary**.

### Interactive API Docs

FastAPI generates documentation automatically:

- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

---

## 📡 API Reference

### `GET /`

Serves the web interface.

### `POST /summarize`

Generates a summary for the given text.

**Request body**

```json
{
  "dialogue": "Amanda: I baked cookies. Do you want some?\nJerry: Sure!\nAmanda: I'll bring you some tomorrow :-)"
}
```

| Field      | Type   | Required | Description                         |
| ---------- | ------ | -------- | ----------------------------------- |
| `dialogue` | string | ✅       | The text or conversation to summarize |

**Response**

```json
{
  "summary": "amanda baked cookies and will bring jerry some tomorrow."
}
```

**cURL**

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"dialogue": "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I will bring you some tomorrow."}'
```

**Python**

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/summarize",
    json={"dialogue": "Amanda: I baked cookies. Do you want some? Jerry: Sure!"},
)
print(response.json()["summary"])
```

---

## 🎛️ Configuration

You can tune generation in the `summarize_dialogue()` function in `main.py`:

| Parameter        | Default | Description                                           |
| ---------------- | ------- | ----------------------------------------------------- |
| `max_length` (input)  | `512`   | Maximum number of input tokens. Longer text is truncated. |
| `max_length` (output) | `150`   | Maximum length of the generated summary, in tokens.  |
| `num_beams`      | `4`     | Beam search width. Higher values may improve quality but are slower. |
| `early_stopping` | `True`  | Stops generation once all beams reach an end token.  |

**Device selection** is automatic, in this priority order:
`MPS (Apple Silicon)` → `CUDA (NVIDIA GPU)` → `CPU`

---

## 🧯 Troubleshooting

| Problem | Solution |
| ------- | -------- |
| `OSError: ./saved_T5_summary_model does not appear to have a file named config.json` | The model folder is missing or incomplete. See [Model Setup](#4-model-setup). |
| `ImportError: T5Tokenizer requires the SentencePiece library` | Run `pip install sentencepiece`. |
| Slow responses | Inference on CPU is slower. Use a GPU or MPS, or reduce `num_beams`. |
| `Address already in use` | Run on another port: `uvicorn main:app --port 8001`. |

---

## 🔮 Future Improvements

- [ ] Adjustable summary length from the UI
- [ ] Upload `.txt` / `.pdf` files for summarization
- [ ] Chunking support for documents longer than 512 tokens
- [ ] Docker support for one-command deployment
- [ ] Deploy to Hugging Face Spaces / Render / AWS
- [ ] Unit and integration tests

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers): pretrained models and tooling
- [T5 paper](https://arxiv.org/abs/1910.10683): *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (Raffel et al.)
- [FastAPI](https://fastapi.tiangolo.com/): the web framework

---

<p align="center">⭐ If you found this project useful, consider giving it a star!</p>
