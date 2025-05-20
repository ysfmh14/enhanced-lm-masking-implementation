import re
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def apply_cleaning(df):
    df["text"] = df["text"].apply(clean_text)
    return df