### BioLLM + RAG Chatbot

A Biomedical Question-Answering Chatbot combining **LLaMA 2** (finetuned with LoRA adapters) and **Retrieval-Augmented Generation (RAG)** using by **Chroma vector store** and **LlamaIndex**. This project tr precise, context-rich biomedical Q\&A through a Gradio interface. Project is deployed on <a href="https://huggingface.co/spaces/nemheen/biomedicalllm">HuggingFace Space.(It's currently puased due to gpu subscription fee)</a>

---

###Overview

This chatbot pipeline includes:

* ✅ LLaMA 2 (7B Chat) loaded in 4-bit with PEFT (LoRA) adapter
* ✅ Biomedical documents indexed with ChromaDB + HuggingFace embeddings
* ✅ Contextual retrieval with LlamaIndex
* ✅ Re-ranking via SentenceTransformer cross-encoder** to filter contexts
* ✅ Lightweight and fast inference using quantization with BitsandBytes
* ✅ Interface through Gradio Chat

---
###Requirements

* Python >= 3.9
* GPU-enabled

For dependencies&libraries:

Refer to requirements.txt

---

### Setup & Execution

#### Step 1: Add your Hugging Face & OpenAI tokens

```python
import os
os.environ["HUGGINGFACE_TOKEN"] = "your_hf_token"
os.environ["OPENAI_API_KEY"] = "your_openai_key"
```

#### Step 2: Download and extract model + index files

The code handles automatic download from HuggingFace if not already present:

```python
# Files:
# - checkpoint.zip (LoRA weights)
# - index.zip      (LlamaIndex storage)
# - chroma_db.zip  (Chroma vector DB)
```

#### Step 3: Load the model + tokenizer

```python
base_model_id = "meta-llama/Llama-2-7b-chat-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    quantization_config=bnb_config,
    token=huggingface_token
)

model_from_checkpoint = PeftModel.from_pretrained(
    base_model,
    adapter_path,  # e.g., "llama/checkpoint-3000"
    token=huggingface_token
)
```

#### Step 4: Launch Gradio Chat Interface

```python
demo.launch()
```

You’ll see a nice web UI that supports:

* Input question
* System message
* Temperature, top-p
* Token limit

---

### 💬 Sample Prompt

> **Q:** What is the mechanism of action of aspirin?

The system retrieves relevant biomedical passages and constructs a prompt like:

```
You are a biomedical expert. Use the given context to answer the question in a concise and clear manner.

Context:
[...] (retrieved docs)

Question: What is the mechanism of action of aspirin?
Answer:
```

---

### 📦 Models Used

* **Base LLM**: `meta-llama/Llama-2-7b-chat-hf` (4-bit)
* **LoRA Checkpoint**: Custom-trained biomedical adapter
* **Embedding Model**: `BAAI/bge-large-en`
* **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

### 🧠 Features

* Retrieval-augmented generation (RAG)
* Biomedical QA fine-tuned adapter
* Fast inference via quantization + LoRA
* Plug-and-play with Hugging Face Hub
* Interactive UI via Gradio

---

### TODO

* [ ] Add streaming responses for long outputs
* [ ] Integrate document upload for dynamic indexing
* [ ] Add support for citation highlighting in output
* [ ] Optimize memory usage for colab-level hardware

---

### 📄 License

MIT License. See `LICENSE.md` for details.

---

### Acknowledgments

* Meta AI for LLaMA 2
* HuggingFace for transformers + PEFT
* LlamaIndex + ChromaDB
* BAAI + MS MARCO teams

### 📝 Reference

If you use or build upon this project, please cite the original paper that inspired it:

> BioinspiredLLM: Conversational Large Language Model for the Mechanics of Biological and Bio-Inspired Materials 
> DOI: [10.1002/advs.202306724](https://doi.org/10.1002/advs.202306724)

