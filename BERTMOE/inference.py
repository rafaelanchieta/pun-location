import os
import torch
import joblib
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report

from .config import CONFIG, DEVICE
from .model import BertMoEClassifier, MoEEnsemble

def load_ensemble_from_saved_models(config, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    models = []

    for seed in config['ensemble_seeds']:
        model_path = os.path.join(config['output_dir'], f'moe_model_seed{seed}.pt')
        if not os.path.exists(model_path): continue

        model = BertMoEClassifier(
            model_name=config['model_name'], num_labels=config['num_labels'],
            num_experts=config['num_experts'], top_k=config['top_k'],
            expert_hidden_dim=config['expert_hidden_dim'], dropout=config['dropout'],
            load_balance_loss_coef=config['load_balance_loss_coef'],
            label_smoothing=config['label_smoothing'], concat_layers=config['concat_last_n_layers'],
            use_focal_loss=config['use_focal_loss'], focal_alpha=config['focal_alpha'],
            focal_gamma=config['focal_gamma']
        )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        models.append(model)
        
    return MoEEnsemble(models=models, method=config['ensemble_method']), tokenizer

def predict_text(text, ensemble, tokenizer, device, threshold=0.5):
    encoding = tokenizer(
        text, max_length=CONFIG['max_length'], padding='max_length', 
        truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        predictions, probabilities = ensemble.predict(
            encoding['input_ids'].to(device), encoding['attention_mask'].to(device), threshold
        )

    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0].cpu().numpy())
    pun_tokens, pun_indices = [], []
    for i, (token, pred, prob) in enumerate(zip(tokens, predictions[0], probabilities[0])):
        if token in ['[CLS]', '[SEP]', '[PAD]']: continue
        if pred == 1:
            pun_tokens.append({
                'token': token, 'index': i, 
                'probability': prob.item() if len(prob.shape) == 0 else prob[1].item()
            })
            pun_indices.append(i)

    return {'texto_original': text, 'tokens_pun': pun_tokens, 'tem_trocadilho': len(pun_tokens) > 0}

def run_hybrid_pipeline(test_data, sklearn_model_path='ensemble_model.pkl'):
    ensemble_data = joblib.load(sklearn_model_path)
    sklearn_model, vectorizer = ensemble_data['model'], ensemble_data['vectorizer']
    ensemble, tokenizer = load_ensemble_from_saved_models(CONFIG, device=DEVICE)
    
    results = []
    for example in tqdm(test_data, desc="Processando Pipeline Híbrido"):
        tokens, labels = example['tokens'], example['labels']
        text = " ".join(tokens)
        
        pun_prob = sklearn_model.predict_proba(vectorizer.transform([text]))[0][1]
        sent_pred = 1 if pun_prob >= 0.4 else 0
        pred_labels, token_probs = [0] * len(tokens), [0.0] * len(tokens)
        
        if sent_pred == 1:
            enc = tokenizer(tokens, is_split_into_words=True, max_length=CONFIG['max_length'], padding='max_length', truncation=True, return_tensors='pt')
            with torch.no_grad():
                bert_preds, bert_probs_tensor = ensemble.predict(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE))
            
            bert_preds, bert_probs = bert_preds[0].cpu().numpy(), bert_probs_tensor[0].cpu().numpy()
            word_ids, prev_word_idx = enc.word_ids(batch_index=0), None
            
            for idx, word_idx in enumerate(word_ids):
                if word_idx is not None and word_idx != prev_word_idx:
                    pred_labels[word_idx] = int(bert_preds[idx])
                    token_probs[word_idx] = float(bert_probs[idx])
                prev_word_idx = word_idx
                
        results.append({
            'text': text, 'tokens': tokens, 'true_labels': labels, 'pred_labels': pred_labels,
            'predicted_probs': [round(p, 4) for p in token_probs], 'stage1_pred': sent_pred, 'stage1_prob': float(pun_prob)
        })
        
    all_true_labels = []
    all_pred_labels = []
    for res in results:
        all_true_labels.extend(res['true_labels'])
        all_pred_labels.extend(res['pred_labels'])
        
    print("\n--- Relatório de Classificação (Nível de Token) ---")
    print(classification_report(all_true_labels, all_pred_labels))
    
    return results