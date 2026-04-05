import torch
from transformers import AutoTokenizer, AutoModel


def pool(hidden_state, mask, pooling_method="mean") -> torch.Tensor:
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]


def init():
    tokenizer = AutoTokenizer.from_pretrained("sergeyzh/rubert-mini-frida")
    model = AutoModel.from_pretrained("sergeyzh/rubert-mini-frida")
    model.eval()
    return tokenizer, model


def tokenize(inputs, tokenizer, truncation=True):
    tokenized_inputs = tokenizer(
        inputs,
        max_length=512 if truncation else None,
        truncation=truncation,
        return_tensors="pt"
    )
    return tokenized_inputs


def predict(tokenized_inputs, model):
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    
    embeddings = pool(
        outputs.last_hidden_state, 
        tokenized_inputs["attention_mask"],
        pooling_method="mean",
    )

    return embeddings.detach().cpu().numpy().tolist()
