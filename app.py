import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model (runs on startup – fast for 3.8B)
@st.cache_resource
def load_model():
    model_name = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False,
        low_cpu_mem_usage=True
    )
    return model, tokenizer

model, tokenizer = load_model()

st.title("Kyzen's Phi-3.5 Mini Chat")
st.write("Day 1 · 8-year-old laptop · 30-day monk mode to top-100 AI engineer. Powered by Microsoft Phi-3.5-mini-instruct.")

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
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,  # Short for speed
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            reply = full_output[len(input_text):].strip()
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})