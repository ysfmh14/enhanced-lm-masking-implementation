# tokenizer_utils.py

from transformers import BertTokenizer

def init_tokenizer(model, additional_tokens=None):
    tokenizer = BertTokenizer.from_pretrained(model)
    if additional_tokens:
        tokenizer.add_tokens(additional_tokens)
    return tokenizer

def bert_tokenize(text, tokenizer):
    return tokenizer.tokenize(text)

def tokenize_function(examples, max_length):
    tokenizer = init_tokenizer
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    return tokenized