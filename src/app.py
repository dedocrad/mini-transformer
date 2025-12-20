import streamlit as st
from model import MiniTransformer
from tokenizer import CharTokenizer
import string
import torch
from stuff import predict

st.title("Mini-Transformer string reverser")

tokenizer = CharTokenizer(string.ascii_letters)
model = MiniTransformer(vocab_size=tokenizer.vocab_size)

model.load_state_dict(torch.load("../models/best_model.pth"))
model.eval()

prompt = st.text_input("Enter a prompt:", "example")
if st.button("Generate Text"):
    generated_text = predict(model, tokenizer, prompt)
    st.success(generated_text)