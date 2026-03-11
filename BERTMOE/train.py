import os
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

from .config import set_seed, CONFIG, DEVICE
from .model import BertMoEClassifier, MoEEnsemble

def train_single_model(config, train_loader, val_loader, device, seed, model_idx):
    set_seed(seed)
    
    model = BertMoEClassifier(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        num_experts=config['num_experts'],
        top_k=config['top_k'],
        expert_hidden_dim=config['expert_hidden_dim'],
        dropout=config['dropout'],
        load_balance_loss_coef=config['load_balance_loss_coef'],
        label_smoothing=config['label_smoothing'],
        concat_layers=config['concat_last_n_layers'],
        use_focal_loss=config['use_focal_loss'],
        focal_alpha=config['focal_alpha'],
        focal_gamma=config['focal_gamma']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=len(train_loader) * config['epochs'])

    best_val_f1 = -1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(config['epochs']):
        model.train()
        all_preds, all_labels = [], []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=-1)
                mask = labels != -100
                val_preds.extend(preds[mask].cpu().numpy())
                val_labels.extend(labels[mask].cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        print(f"\nEpoch {epoch+1} - Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            print(f"Melhoria no F1: {best_val_f1:.4f} -> {val_f1:.4f}. Salvando checkpoint!")
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(config['output_dir'], f'moe_model_seed{seed}.pt'))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Sem melhoria (Paciência: {epochs_without_improvement}/{config['early_stopping_patience']})")

        if epochs_without_improvement >= config['early_stopping_patience']:
            break

    model.load_state_dict(best_state)
    return model, best_val_f1

def train_ensemble(train_loader, val_loader):
    models, val_f1s = [], []
    for i, seed in enumerate(CONFIG['ensemble_seeds']):
        model, val_f1 = train_single_model(CONFIG, train_loader, val_loader, DEVICE, seed, i)
        models.append(model)
        val_f1s.append(val_f1)
        torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], f'moe_model_seed{seed}.pt'))

    weights = val_f1s if CONFIG['ensemble_method'] == 'weighted' else None
    return MoEEnsemble(models=models, method=CONFIG['ensemble_method'], weights=weights)