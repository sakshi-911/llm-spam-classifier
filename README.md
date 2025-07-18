# 📬 SpamGPT - Spam Classification using LLM

SpamGPT is a lightweight spam detection application powered by a custom GPT2-style transformer model. It classifies user input as either **Spam** or **Not Spam**, using a fine-tuned language model with a minimal interface built in Streamlit.

[spamgpt.streamlit.app](https://spamgpt.streamlit.app/)

---

## 🚀 Features

- ✅ Lightweight GPT model trained from scratch
- ✅ Fast and accurate spam classification
- ✅ Built with Streamlit for a simple web UI
- ✅ Deployable to Streamlit Cloud
- ✅ Hugging Face integration for model weights

---

## 🧠 Model Details

- Architecture: Custom GPT-style Transformer
- Embedding size: Configurable
- Training: Fine-tuned on labeled spam datasets
- Final Layer: 2-class Linear head

The model is loaded using:

```python
def load_model():
    model = gpt_model(cfg)
    model.out_head = torch.nn.Linear(cfg["emb_dim"], 2)
    model.load_state_dict(torch.load("spam_classifier.pth", map_location="cpu"))
    model.eval()
    return model


```
## 🖥️ Run Locally

```bash
git clone https://github.com/sakshi-911/llm-spam-classifier.git
cd llm-spam-classifier
conda create -n spam-env python=3.10 -y
conda activate spam-env
pip install -r requirements.txt
streamlit run app.py
```



### Since large models can’t be pushed to GitHub, the .pth file is hosted on Hugging Face:
```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="sakshi-911/spam-classifier",
    filename="spam_classifier.pth",
    local_dir="./checkpoints"
)
```



```text
📁 Project Structure
.
├── app.py                          # Streamlit UI application
├── functions/
│   ├── classify.py                 # Handles spam classification logic
│   ├── convert.py                  # Utilities for data/token conversion
├── model/
│   ├── feed_forward.py         # Feed-forward network used in Transformer blocks
│   ├── gelu.py                 # GELU activation function implementation
│   ├── gpt_model.py            # Main GPT model class definition
│   ├── layer_norm.py           # Layer normalization module
│   ├── multihead_attention.py  # Multi-head self-attention mechanism
│   ├── transformer.py          # Transformer block combining attention and feed-forward layers
├── model_config/                  # Configuration files for the model
│   ├── model_config.py
├── requirements.txt               # Python dependencies
├── spam_classifier.pth           # Trained model weights (ignored in github)
├── upload.py                      # File upload handling logic
├── .gitignore                     # Git ignore rules
└── llm-spam-classifier.code-workspace  # VS Code workspace file
```

