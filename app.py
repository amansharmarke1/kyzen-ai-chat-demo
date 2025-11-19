import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load Granite-4.0-H-350M in INT8 (fits free tier RAM)
@st.cache_resource
def load_model():
    model_name = "ibm-granite/granite-4.0-h-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # INT8 quantization config (halves memory)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["lm_head"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,  # INT8 for low memory
        device_map="cpu",  # Force CPU to avoid GPU conflicts
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16  # Fix deprecation
    )
    return model, tokenizer

model, tokenizer = load_model()

st.title("Kyzen's Granite-4.0-H-350M Chat")
st.write("Day 1 · 8-year-old laptop · 30-day monk mode to top-100 AI engineer. Powered by IBM Granite-4.0-H-350M (INT8 quantized for free tier).")

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
            # Granite prompt template
            input_text = f"<s>[INST] {prompt} [/INST]"
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,  # Even shorter for 1–3s replies
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            reply = full_output[len(input_text):].strip()
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})