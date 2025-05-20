# enhanced-lm-masking-implementation

This repository provides the implementation of the training pipeline introduced in the paper:

**"Enhancing Language Models for Specialized Classification Tasks Using Selective Masking Strategies" **

## 📄 Abstract

This paper introduces a training pipeline for pre-trained language models, consisting of two phases. The first phase involves post-training guided by selective masking to adapt the model to a specific domain, while the second phase is reserved for fine-tuning the model on the classification task. The objective is to address the problem of scarce labeled data in specialized domains such as biomedical, plant health, and animal health. Furthermore, we propose two novel selective masking strategies designed for the post-training phase. We performed several experiments to evaluate the performance of our approach using the proposed selective masking strategies, across three specific domains: biomedical, plant health, and syndromic surveillance, as well as with three language models: BERTBase, SciBERT, and BioBERT. The results demonstrated significant improvements in the performance of the language models in the classification task. Additionally, we proposed a summary that maps each strategy to the context in which it is most effective.

---

## 🧩 Project Structure

├── main.py # Main pipeline: post-training + fine-tuning
├── scripts/
│ ├── cleaning.py # Text cleaning utilities
│ ├── config.py # Config files and constants
│ ├── data_collator.py # Custom data collators for two proposed strategies
│ ├── data_loader.py # Load datasets and lexicons
│ ├── metrics.py # Evaluation metrics
│ ├── tfidf_matrix.py # TF-IDF computation for masking
│ └── tokenizer_utils.py # Tokenizer initialization and processing
