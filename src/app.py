import streamlit as st
from model import MiniTransformer
from tokenizer import CharTokenizer
import string
import torch
from stuff import predict
import matplotlib.pyplot as plt
import mlflow

st.title("Mini-Transformer string reverser")

tokenizer = CharTokenizer(string.ascii_letters)
model = MiniTransformer(vocab_size=tokenizer.vocab_size)

model.load_state_dict(
    torch.load("models/best_model.pth", map_location="cpu"))

model.eval()

prompt = st.text_input("Enter a prompt:", "example")
INPUT_IDS = tokenizer.encode(prompt)
LEN_PROMPT = len(INPUT_IDS)
if st.button("Generate Text"):
    generated_text = predict(model, tokenizer, prompt)
    st.success(generated_text)

    target_text = prompt[::-1]
    is_correct = generated_text == target_text
    score = 1.0 if is_correct else 0.0

    if is_correct:
        st.success("Correct reversal")
    else: st.error("Incorrect reversal")

    with mlflow.start_run(run_name="inference"):
        mlflow.log_param("input_prompt", prompt)
        mlflow.log_param("target", target_text)
        mlflow.log_param("generated", generated_text)
        mlflow.log_param("exact_match", score)