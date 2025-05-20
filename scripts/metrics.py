from sklearn.metrics import accuracy_score
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)

    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
    }