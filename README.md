# enhanced-lm-masking-implementation

This repository provides the implementation of the training pipeline introduced in the paper:

***Enhancing Language Models for Specialized Classification Tasks Using Selective Masking Strategies***

## 📄 Abstract

This paper introduces a training pipeline for pre-trained language models, consisting of two phases. The first phase involves post-training guided by selective masking to adapt the model to a specific domain, while the second phase is reserved for fine-tuning the model on the classification task. The objective is to address the problem of scarce labeled data in specialized domains such as biomedical, plant health, and animal health. Furthermore, we propose two novel selective masking strategies designed for the post-training phase. We performed several experiments to evaluate the performance of our approach using the proposed selective masking strategies, across three specific domains: biomedical, plant health, and syndromic surveillance, as well as with three language models: BERTBase, SciBERT, and BioBERT. The results demonstrated significant improvements in the performance of the language models in the classification task. Additionally, we proposed a summary that maps each strategy to the context in which it is most effective.

---

## 🧩 Project Structure

    selective-masking-mlm-reproduction/ 
        ├── main.py
        ├── requirements.txt
        ├── README.md
        └── scripts/    
            ├── cleaning.py
            ├── config.py
            ├── data_collator.py
            ├── data_loader.py
            ├── metrics.py
            ├── tfidf_matrix.py
            └── tokenizer_utils.py


---

## 🚀 How to Run

 ### 1. Clone the Repository

    git clone https://github.com/your-username/selective-masking-mlm-reproduction.git
    cd selective-masking-mlm-reproduction


### 2. Install Dependencies
Make sure you have Python 3.8+ installed, then run:

    pip install -r requirements.txt
### 3. Prepare Data

Place your datasets and lexicon files in the appropriate format, and update the paths in `scripts/config.py`:

    - `POST_TRAINING_DATA_FILE`
    - `FINE_TUNING_DATA_FILE`
    - `LEXIC_FILE`
### 4. Run the Pipeline

    python main.py


---

## ⚙️Proposed Pipeline and Selective masking strategies:
 
  ### Pipeline
![Aperçu du pipeline](images/Pipline_training.png)
  ### Strategy 1
![Aperçu du pipeline](images/strategy2-vf1.png)
  ### Strategy 2
![Aperçu du pipeline](images/strategy1-vF2.png)

## 📁 Citation
If you use this code, please cite the original paper:

***Enhancing Language Models for Specialized Classification Tasks Using Selective Masking Strategies.***

## 📬 Contact
If you have any questions, encounter issues with the code, or would like to know more about our work, please contact the corresponding author:

📧 Personal email (permanent): ysfmh2002@gmail.com

📧 Professional email (not sure if permanent): mahdoubi.youssef@usms.ma
