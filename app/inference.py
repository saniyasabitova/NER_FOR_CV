
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import torch


model_id = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)

MODEL_PATH = "models/xlm_roberta_ner"

model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()


def predict_chunk(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    return probabilities


def chunk_tokens(text, max_tokens: int, overlap: int):
    tokens = tokenizer.tokenize(text)
    chunks = []

    start = 0

    while start < len(tokens):
        end = start + max_tokens

        chunk = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_text)

        if end >= len(tokens):
            break

        start += max_tokens - overlap

    return chunks

