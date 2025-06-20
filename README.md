🧬 BioLLM + RAG Chatbot

This is a biomedical question-answering chatbot built using LLaMA 2 (fine-tuned with LoRA adapters) and a Retrieval-Augmented Generation (RAG) pipeline powered by Chroma vector store and LlamaIndex. It’s designed to help people in the biology field ask questions and explore scientific content by bringing papers from various sources into one place.

While the current document database is still limited and the RAG results aren’t perfect, the system can improve as more scientific texts are added to the vector database.

The project is deployed on <a href="https://huggingface.co/spaces/nemheen/biomedicalllm">Hugging Face Spaces</a> (currently paused due to GPU subscription costs).
This is a solo project I built myself, inspired by the paper BioinspiredLLM: Conversational Large Language Model for the Mechanics of Biological and Bio-Inspired Materials.


---

📦 Key Files (Downloadable from Hugging Face Hub)

checkpoint.zip
Contains the LoRA adapter weights from finetuning LLaMA 2 on a custom biomedical QA dataset.

Trained on 22,000 biomedical QA pairs

3000 training steps using transformers.Trainer

Optimizer: AdamW with learning rate = 2e-4, epsilon = 1e-8

Gradient clipping: 0.3

Parameters were reduced by 99.2586% using LoRA

Training logs tracked using Weights & Biases (wandb)

Note: Model resumes from step 500 during initialization to skip warm-up


index.zip
Stores the LlamaIndex vector index, built on top of biomedical documents. It handles fast semantic search over the context chunks used in RAG.

chroma_db.zip
This is the Chroma vector database, which stores embedded biomedical documents using BAAI/bge-large-en embeddings. It’s the source for relevant documents during retrieval.



---

📋 Overview

This chatbot pipeline includes:

- LLaMA 2 (7B Chat) loaded in 4-bit with PEFT (LoRA) adapter

- Biomedical documents fetched, cleaned, indexed with ChromaDB + HuggingFace embeddings

- Contextual retrieval with LlamaIndex

- Re-ranking via SentenceTransformer cross-encoder to filter top-relevant contexts

- Fast, memory-efficient inference using quantization (via BitsandBytes)

- Interactive web chat using Gradio



---

⚙️ Requirements

Python >= 3.9

GPU-enabled runtime (CUDA)


To install all required dependencies, refer to requirements.txt.


---

🚀 Setup & Execution

Step 1: Add your API tokens

import os
os.environ["HUGGINGFACE_TOKEN"] = "your_hf_token"
os.environ["OPENAI_API_KEY"] = "your_openai_key"

Step 2: Unzip and configure paths

Download the following files and unzip:

checkpoint.zip → LoRA adapter weights

index.zip → LlamaIndex storage

chroma_db.zip → Chroma vector DB


Make sure to configure their paths x
correctly in app.py.

Step 3: Run the app on huggingface space. (or main_biollm.ipynb)

Launch the Gradio interface:

You’ll get a web UI with:

Input field for biomedical questions

Adjustable parameters (temperature, top-p, token limit)

System message area for persona prompts



---

💬 Sample Prompt

> Q: What is the mechanism of action of aspirin?



The system retrieves relevant biomedical passages and creates a contextual prompt like this:

You are a biomedical expert. Use the given context to answer the question in a concise and clear manner.

Context:
[...] (retrieved docs)

Question: What is the mechanism of action of aspirin?
Answer:


---

🔍 Models Used

Component	Model

Base LLM	meta-llama/Llama-2-7b-chat-hf (4-bit)
LoRA Checkpoint	Custom-trained biomedical adapter (checkpoint.zip)
Embedding Model	BAAI/bge-large-en
Reranker	cross-encoder/ms-marco-MiniLM-L-6-v2



---

🧠 Features

Retrieval-Augmented Generation (RAG)

Biomedical QA adapter (LoRA-based)

Efficient inference (quantized LLaMA + BitsAndBytes)

Plug-and-play design (via Hugging Face + Gradio)



---

✅ TODO

[ ] Add streaming responses for long outputs

[ ] Allow PDF/document upload for dynamic indexing

[ ] Add citation markers for retrieved context in responses

[ ] Optimize for Colab-level hardware



---

📄 License

MIT License. See LICENSE.md for full terms.


---

🙏 Acknowledgments

Meta AI for LLaMA 2

Hugging Face for Transformers, PEFT, and model hosting

LlamaIndex + ChromaDB for vector-based search

BAAI and MS MARCO for embeddings and reranker models



---

📝 Reference

If you use or build upon this project, please cite the original paper that inspired it:

> BioinspiredLLM: Conversational Large Language Model for the Mechanics of Biological and Bio-Inspired Materials
> DOI: 10.1002/advs.202306724
