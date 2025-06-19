from huggingface_hub import hf_hub_download
import os
import requests
import zipfile

openai_key = os.getenv("OPENAI_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"



base_dir = "nemheen/biomedicalllm/resolve/main"
os.makedirs(base_dir, exist_ok=True)

# URLs dict
urls = {
    "llama": "https://huggingface.co/spaces/nemheen/biomedicalllm/resolve/main/checkpoint.zip",
    "index": "https://huggingface.co/spaces/nemheen/biomedicalllm/resolve/main/index.zip",
    "chroma": "https://huggingface.co/spaces/nemheen/biomedicalllm/resolve/main/chroma_db.zip"
}

for name, url in urls.items():
    zip_path = os.path.join(base_dir, f"{name}.zip")
    extract_path = os.path.join(base_dir, name)

    # Download if not exist
    if not os.path.exists(zip_path):
        print(f"Downloading {name} from {url}...")
        r = requests.get(url)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)
        print(f"{name} downloaded to {os.path.abspath(zip_path)}")
    else:
        print(f"{name} already downloaded at {os.path.abspath(zip_path)}")

    # Extract if not exist
    if not os.path.exists(extract_path):
        print(f"Extracting {name} to {os.path.abspath(extract_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"{name} extracted to: {os.path.abspath(extract_path)}")

        print(f"All files and folders under {os.path.abspath(extract_path)}:")
        for root, dirs, files in os.walk(extract_path):
            for d in dirs:
                print("DIR :", os.path.join(root, d))
            for f in files:
                print("FILE:", os.path.join(root, f))
    else:
        print(f"{name} already extracted at: {os.path.abspath(extract_path)}")

chroma_dir = os.path.join(base_dir, "chroma")
index_dir = os.path.join(base_dir, "index")
model_dir = os.path.join(base_dir, "llama")


persist_dir = "/home/user/app/nemheen/biomedicalllm/resolve/main/chroma/chroma_db_final"
index_dir = "/home/user/app/nemheen/biomedicalllm/resolve/main/index/index"
fine_tuned_checkpoint = "/home/user/app/nemheen/biomedicalllm/resolve/main/llama/checkpoint-3000"

 

embedding_model_name = "BAAI/bge-large-en"





from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

import gradio as gr
import torch
import time
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter
from peft import PeftConfig, PeftModel

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Base model from Hugging Face
base_model_id = "meta-llama/Llama-2-7b-chat-hf"
adapter_path = "nemheen/biomedicalllm/resolve/main/llama/checkpoint-3000"


# Configuratiion of 4-bit quantization using bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_compute_dtype=torch.float16  
)

# Load model with 4-bit quantization
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    quantization_config=bnb_config,
    token=huggingface_token
)

model_from_checkpoint = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    token=huggingface_token
)


tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=huggingface_token)
tokenizer.pad_token = tokenizer.eos_token  

# print(model_from_checkpoint.hf_device_map)  # Show which devices the model is loaded on
print(model_from_checkpoint.dtype)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_from_checkpoint.to(device)




# BioLLM wrapper
class BioLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def complete(self, prompt, max_tokens=100, **generate_kws):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            # do_sample=False,
            **generate_kws,
            pad_token_id=self.tokenizer.eos_token_id
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded


bio_llm = BioLLM(model=model_from_checkpoint, tokenizer=tokenizer)



Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
Settings.num_output = 312
Settings.context_window = 3900
# Settings.llm = llm


# Load Chroma vector, index store
embedding_model = HuggingFaceEmbedding(model_name=embedding_model_name)
chroma_vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedding_model,
)
chroma_collection = chroma_vectorstore._collection
chroma_vectorstore = ChromaVectorStore(chroma_collection=chroma_collection)

# Load LlamaIndex from storage
storage_context = StorageContext.from_defaults(
    persist_dir=index_dir,
    vector_store=chroma_vectorstore
)
index = load_index_from_storage(storage_context)

# Set up retriever and reranker
reranker = SentenceTransformerRerank(
    top_n=2,
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    keep_retrieval_score=False
)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
    node_postprocessors=[reranker]
)

from huggingface_hub import InferenceClient
import gradio as gr

# Assume bio_llm and retriever are defined and loaded before this
# from your_model_module import bio_llm, retriever

def chat_fn(message, system_message=None, max_tokens=256, temperature=0.0, top_p=0.95):

    try:
        print(f"[DEBUG] Incoming message: {message}")
        print(f"[DEBUG] Type of message: {type(message)}")

        # Step 1: Retrieve context
        retrieved_nodes = retriever.retrieve(message)
        context = "\n\n".join([node.text for node in retrieved_nodes])
        print(f"[DEBUG] Retrieved {len(retrieved_nodes)} nodes")

        # Step 2: Format prompt
        prompt = f"""You are a biomedical expert. Use the given context to answer the question in a concise and clear manner.

Context:
{context}

Question: {message}
Answer:"""

    
        start_time = time.time()
        raw_response = bio_llm.complete(prompt, 
                                        max_tokens=max_tokens, 
                                        do_sample=True if temperature > 0.0 else False,
                                        temperature=temperature,
                                        top_p=top_p,
                                       ).strip()
        print(f"[DEBUG] Raw response: {raw_response}")
        end_time = time.time()
        elapsed = end_time - start_time
        
        if "answer:" in raw_response.lower():
            idx = raw_response.lower().find("answer:")
            cleaned = raw_response[idx + len("answer:"):].strip()
        else:
            cleaned = raw_response.strip()

        print(cleaned)
        # if cleaned:
        #     cleaned = cleaned[0].upper() + cleaned[1:]
        cleaned += f"\n\n⏱️ Generated in {elapsed:.2f} seconds."

        return cleaned
        
    except Exception as e:
        import traceback
        print("[ERROR] An exception occurred in chat_fn:")
        traceback.print_exc()
        return f"⚠️ Error: {str(e)}"

def respond(message, history, system_message, max_tokens, temperature, top_p):
    try:
        response = chat_fn(
        message,
        system_message=system_message,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
        )
        history.append((message, response))
        yield response
    
    except Exception as e:
        import traceback
        print("[ERROR] Exception in respond:")
        traceback.print_exc()
        yield f"⚠️ Error: {str(e)}"


# Build ChatInterface with extra parameters
demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a biomedical expert.", label="System message"),
        gr.Slider(minimum=1, maximum=512, value=256, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
    title="🧬 BioLLM + RAG Chatbot",
    description="Ask biomedical questions and receive precise, context-aware answers.",
    theme=gr.themes.Soft(),
)

demo.launch()
