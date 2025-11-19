import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_model()

st.title("Kyzen's Day-1 Chatbot")
st.write("Live on Streamlit free tier · Replies in 1–2 seconds · Zero crashes")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
            reply_ids = model.generate(inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
            reply = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})