
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizer_utils import bert_tokenize

def compute_tfidf(texts, tokenizer):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        tokenizer= bert_tokenize

    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    tokens = vectorizer.get_feature_names_out()
    return tfidf_matrix, tokens
