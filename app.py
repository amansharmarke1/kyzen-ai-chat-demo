import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch

# Load Granite-4.0-H-350M in FP16 (fits 1 GB RAM)
@st.cache_resource
def load_model():
    model_name = "ibm-granite/granite-4.0-h-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    return model, tokenizer

model, tokenizer = load_model()

st.title("Kyzen's Granite-4.0-H-350M Chat")
st.write("Day 1 · 8-year-old laptop · 30-day monk mode to top-100 AI engineer. Powered by IBM Granite-4.0-H-350M (streaming for free tier).")

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

    # Generate reply with streaming (lowers peak RAM)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_reply = ""
        with st.spinner("Thinking..."):
            # Granite prompt template
            input_text = f"<s>[INST] {prompt} [/INST]"
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            # Streaming setup
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=64,  # Super short for 1–3s replies
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )

            # Stream tokens progressively
            for new_text in streamer:
                full_reply += new_text
                message_placeholder.markdown(full_reply + "▌")  # Cursor effect

        message_placeholder.markdown(full_reply)
    st.session_state.messages.append({"role": "assistant", "content": full_reply})