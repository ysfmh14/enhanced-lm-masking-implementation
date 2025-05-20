# main.py

from scripts.config import POST_TRAINING_DATA_FILE,FINE_TUNING_DATA_FILE, LEXIC_FILE, MODEL_NAME,LABEL_NUM, USED_STRATEGY,MAX_LENGTH_TEXT
from scripts.data_loader import load_data, load_lexicon
from scripts.cleaning import apply_cleaning
from scripts.tokenizer_utils import init_tokenizer, tokenize_function
from scripts.tfidf_matrix import compute_tfidf
from scripts.data_collator import Strategy1DataCollator,Strategy2DataCollator
from datasets import Dataset
from scripts.metrics import compute_metrics
from functools import partial
from transformers import BertForMaskedLM, Trainer,BertForSequenceClassification, TrainingArguments
from sklearn.model_selection import train_test_split

def filter_multiword_lexicon(df):
    return df[df['Name'].str.split().str.len() > 1]['Name'].tolist()

def main():


    #Post-training
    df = load_data(POST_TRAINING_DATA_FILE)
    df_lexicon = load_lexicon(LEXIC_FILE)
    liste_lexicon = df_lexicon["Name"].tolist()
    df = apply_cleaning(df)
    list_filtred_lexicon = filter_multiword_lexicon(liste_lexicon)
    tokenizer = init_tokenizer(list_filtred_lexicon, MODEL_NAME)
    tfidf_matrix, tfidf_tokens = compute_tfidf(df["text"], tokenizer)
    unsupervised_dataset = Dataset.from_dict({"text": df["text"].tolist()})
    mlm_model = BertForMaskedLM.from_pretrained("allenai/scibert_scivocab_uncased")
    mlm_model.resize_token_embeddings(len(tokenizer))
    tokenize_for_post_training = partial(tokenize_function, max_length=MAX_LENGTH_TEXT)
    unsupervised_dataset = unsupervised_dataset.map(tokenize_for_post_training, batched=True)
    document_indices = {tuple(example["input_ids"]): idx for idx, example in enumerate(unsupervised_dataset)}
    
    if USED_STRATEGY == 1 :
        data_collator = Strategy1DataCollator(tokenizer, tfidf_matrix, tfidf_tokens,document_indices,liste_lexicon, mask_prob=0.15)
    elif USED_STRATEGY == 2:
        data_collator = Strategy2DataCollator(tokenizer, tfidf_matrix, tfidf_tokens,document_indices, liste_lexicon, mask_prob=0.15)
    else:
        raise ValueError(f"USED_STRATEGY={USED_STRATEGY} is not valid. Use 1 or 2.")
    
    training_args_mlm = TrainingArguments(
            output_dir="./mlm_results",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            num_train_epochs=6,  
            weight_decay=0.01,
            logging_dir='./mlm_logs',
            save_total_limit=2,
            dataloader_pin_memory=False,
            no_cuda=False 
    )
    trainer_mlm = Trainer(
        model=mlm_model,
        args=training_args_mlm,
        train_dataset=unsupervised_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator  
     )
    
    trainer_mlm.train()
    
    mlm_model.save_pretrained("./post_trained_model")
    tokenizer.save_pretrained("./post_trained_model")


    #Fine-tuning
    df_classification = load_data(FINE_TUNING_DATA_FILE)
    df_classification = apply_cleaning(df_classification)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_classification['text'].tolist(), df_classification['label'].tolist(), test_size=0.2, random_state=42
    )
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    tokenizer = init_tokenizer("./post_trained_model")
    tokenize_for_fine_tuning = partial(tokenize_function, max_length=MAX_LENGTH_TEXT)
    train_dataset = train_dataset.map(tokenize_for_fine_tuning, batched=True)
    val_dataset = val_dataset.map(tokenize_for_fine_tuning, batched=True)
    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    model = BertForSequenceClassification.from_pretrained("./post_trained_model", num_labels=LABEL_NUM)
    training_args_classification = TrainingArguments(
            output_dir=".classification_results",
            eval_strategy="epoch",        
            save_strategy="epoch",              
            learning_rate=2e-5,                
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=4,               
            weight_decay=0.01,
            logging_dir='./classification_logs',
            save_total_limit=2,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )
    trainer = Trainer(
        model=model,
        args=training_args_classification,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        )

    trainer.train()

    results = trainer.evaluate()
    print("Accuracy:", results["eval_accuracy"])

if __name__ == "__main__":
    main()
