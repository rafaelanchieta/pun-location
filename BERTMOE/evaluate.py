import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

def evaluate_ensemble(ensemble, data_loader, device, threshold=0.5):
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(data_loader, desc="Avaliando"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        predictions, probabilities = ensemble.predict(input_ids, attention_mask, threshold)

        mask = labels != -100
        all_preds.extend(predictions[mask].cpu().numpy())
        all_labels.extend(labels[mask].cpu().numpy())
        all_probs.extend(probabilities[mask].cpu().numpy())

    return {
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'probabilities': all_probs,
        'labels': all_labels
    }

def optimize_ensemble_threshold(ensemble, data_loader, device):
    all_probs, all_labels = [], []
    for batch in tqdm(data_loader, desc="Otimizando threshold"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        _, probabilities = ensemble.predict(input_ids, attention_mask)
        mask = labels != -100
        all_probs.extend(probabilities[mask].cpu().numpy())
        all_labels.extend(labels[mask].cpu().numpy())

    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    best_f1, best_threshold = 0, 0.5
    for threshold in np.arange(0.30, 0.71, 0.01):
        preds = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold