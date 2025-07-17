import torch
import tiktoken
import streamlit as st
from huggingface_hub import hf_hub_download

from model import gpt_model
from model_config import model_config
from functions import classify, convert

cfg = model_config.GPT_CONFIG_124M
gpt_model = gpt_model.GPTModel
classify_review = classify.classify_review
text_to_token_ids = convert.text_to_token_ids
token_ids_to_text = convert.token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="sakshi-911/spam-classifier", 
        filename="spam_classifier.pth",        
        repo_type="model"                      
    )

    model = gpt_model(cfg)
    model.out_head = torch.nn.Linear(in_features=cfg["emb_dim"], out_features=2)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()
tokenizer = tiktoken.get_encoding("gpt2")

st.title("📩 Spam Classifier")

with st.sidebar:
    st.markdown("## 🔧 Settings")
    max_tokens = st.number_input("Max Tokens to Generate (Not used in classification)", value=50)
    if st.button("🔄 Clear Chat"):
        st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Type a message and hit Enter...")

if user_input:
    st.session_state.history.append(("user", user_input))

    output_text = classify_review(user_input, model, tokenizer, device, max_length=100)
    if output_text =="not spam":
        output_text="Not Spam"
    else:
        output_text="Spam"
    st.session_state.history.append(("bot", output_text))


for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)