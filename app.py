# %% [markdown]
# # Libraries
# %%
import os
import re

import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import BertForSequenceClassification, BertTokenizer

from gdrive_utils import GDriveUtils

# %% [markdown]
# # Config


# %%
from dotenv import load_dotenv
load_dotenv()

MODEL_WEIGHTS_GDRIVE_FILE_ID = os.environ.get("MODEL_WEIGHTS_GDRIVE_FILE_ID")
MODEL_WEIGHTS_LOCAL_PATH = os.environ.get("MODEL_WEIGHTS_LOCAL_PATH", "BERT-imdb-weights.pt")
DOWNLOAD_MODEL_WEIGTHS_FROM_GDRIVE = os.environ.get("DOWNLOAD_MODEL_WEIGTHS_FROM_GDRIVE", "True") == "True"
LOG_GDRIVE_EVENTS = os.environ.get("LOG_GDRIVE_EVENTS", "True") == 'True'


GDriveUtils.LOG_EVENTS = LOG_GDRIVE_EVENTS

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(
        f"{torch.cuda.device_count()} GPU(s) available. Using the GPU: {torch.cuda.get_device_name(0)}"
    )
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Mac ARM64 GPU")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")

# %%
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
if DOWNLOAD_MODEL_WEIGTHS_FROM_GDRIVE:
    GDriveUtils.download_file_from_gdrive(
        MODEL_WEIGHTS_GDRIVE_FILE_ID, MODEL_WEIGHTS_LOCAL_PATH
    )

# %%
model.to(device)
model.device

# %%
model.load_state_dict(torch.load(MODEL_WEIGHTS_LOCAL_PATH, map_location=device))
model.eval()

# %%
def clean_text(x):
    x = re.sub('[,\.!?:()"]', "", x)
    x = re.sub("<.*?>", " ", x)
    x = re.sub("http\S+", " ", x)
    x = re.sub("[^a-zA-Z0-9]", " ", x)
    x = re.sub("\s+", " ", x)
    return x.lower().strip()


# %%
# Tokenize all of the sentences and map the tokens to thier word IDs.
def tokenize_sentences(data):
    input_ids = []
    attention_masks = []

    for sent in data:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict["input_ids"])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict["attention_mask"])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

# %% [markdown]
# # Gradio app

# %%
def process_input(text: str):
    input_ids, attention_masks = tokenize_sentences([text])
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    with torch.no_grad():
        result = model(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_masks,
            return_dict=True,
        )

    return "Positive" if np.argmax(result.logits.cpu().numpy()) == 1 else "Negative"


# Create the Gradio interface
interface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Textbox(label="Enter Movie review"),
    ],
    outputs=[
        gr.Textbox(label="Result"),
    ],
    live=False,  # Optional: updates output as input changes
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
