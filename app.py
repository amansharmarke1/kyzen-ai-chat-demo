import streamlit as st
from llama_cpp import Llama
import torch

# Load quantized GGUF model (4-bit, fast on CPU, <3 GB RAM)
@st.cache_resource
def load_model():
    llm = Llama(
        model_path="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        n_ctx=2048,  # Context length
        n_threads=1,  # Limit threads for free tier (1 CPU)
        n_gpu_layers=0,  # CPU only
        verbose=False
    )
    return llm

llm = load_model()

st.title("Kyzen's Phi-3.5 Mini Chat (Quantized)")
st.write("Day 1 · 8-year-old laptop · 30-day monk mode to top-100 AI engineer. Powered by 4-bit quantized Phi-3.5-mini (fits free tier).")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Phi-3.5 prompt template
            prompt_template = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            
            output = llm(
                prompt_template,
                max_tokens=128,  # Short for speed
                temperature=0.7,
                top_p=0.9,
                echo=False,
                stop=["<|end|>"]
            )
            
            reply = output['choices'][0]['text'].strip()
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})