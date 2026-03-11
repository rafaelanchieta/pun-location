import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, ignore_index=-100, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        mask = targets != self.ignore_index
        logits = logits[mask]
        targets = targets[mask]

        if len(targets) == 0:
            return torch.tensor(0.0, device=logits.device)

        probs = F.softmax(logits, dim=-1)

        if self.label_smoothing > 0:
            num_classes = logits.size(-1)
            smooth_targets = torch.zeros_like(probs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes=logits.size(-1)).float()

        pt = (probs * smooth_targets).sum(dim=-1)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        ce_loss = -torch.log(pt + 1e-8)
        loss = focal_weight * ce_loss
        return loss.mean()

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class Router(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_weights, top_k_indices, probs

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = Router(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout) for _ in range(num_experts)
        ])

    def forward(self, x, return_expert_info=False):
        batch_size, seq_len, input_dim = x.size()
        weights, indices, probs = self.router(x)
        flat_x = x.view(-1, input_dim)
        flat_weights = weights.view(-1, self.top_k)
        flat_indices = indices.view(-1, self.top_k)
        results = torch.zeros_like(flat_x)
        
        expert_outputs_all = {i: torch.zeros_like(flat_x) for i in range(self.num_experts)} if return_expert_info else None

        for k in range(self.top_k):
            expert_indices = flat_indices[:, k]
            expert_weights = flat_weights[:, k].unsqueeze(1)
            for i in range(self.num_experts):
                mask = (expert_indices == i)
                if mask.any():
                    selected_inputs = flat_x[mask]
                    expert_output = self.experts[i](selected_inputs)
                    results[mask] += expert_output * expert_weights[mask]
                    if return_expert_info:
                        expert_outputs_all[i][mask] = expert_output

        output = results.view(batch_size, seq_len, input_dim)
        if return_expert_info:
            expert_info = {
                'weights': weights, 'indices': indices, 'router_probs': probs,
                'expert_outputs': {k: v.view(batch_size, seq_len, -1) for k, v in expert_outputs_all.items()}
            }
            return output, probs, expert_info
        return output, probs

class BertMoEClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2, num_experts=4, top_k=2,
                 expert_hidden_dim=256, dropout=0.35, load_balance_loss_coef=0.1,
                 label_smoothing=0.1, concat_layers=4, use_focal_loss=True, focal_alpha=0.75, focal_gamma=2.0):
        super().__init__()
        self.num_labels = num_labels
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_loss_coef = load_balance_loss_coef
        self.concat_layers = concat_layers

        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.hidden_size = self.bert.config.hidden_size
        self.concat_size = self.hidden_size * concat_layers

        self.layer_projection = nn.Sequential(
            nn.Linear(self.concat_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.moe = MixtureOfExperts(
            input_dim=self.hidden_size, hidden_dim=expert_hidden_dim,
            output_dim=self.hidden_size, num_experts=num_experts, top_k=top_k, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        if use_focal_loss:
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=-100, label_smoothing=label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

    def forward(self, input_ids, attention_mask, labels=None, return_expert_info=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        concat_hidden = torch.cat(hidden_states[-self.concat_layers:], dim=-1)

        sequence_output = self.layer_projection(concat_hidden)
        sequence_output = self.dropout(sequence_output)

        if return_expert_info:
            moe_output, router_probs, expert_info = self.moe(sequence_output, return_expert_info=True)
            expert_info['bert_hidden_states'] = sequence_output
            expert_info['attention_mask'] = attention_mask
        else:
            moe_output, router_probs = self.moe(sequence_output)
            expert_info = None

        moe_output = moe_output + sequence_output
        moe_output = self.layer_norm(moe_output)
        moe_output = self.dropout2(moe_output)
        logits = self.classifier(moe_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            expert_importance = router_probs.mean(dim=(0, 1))
            target = 1.0 / self.num_experts
            balance_loss = torch.mean((expert_importance - target) ** 2)
            loss = loss + self.load_balance_loss_coef * balance_loss

        if return_expert_info:
            return loss, logits, expert_info
        return loss, logits

class MoEEnsemble:
    def __init__(self, models, method='soft_voting', weights=None):
        self.models = models
        self.method = method
        self.weights = weights or [1.0] * len(models)
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def predict(self, input_ids, attention_mask, threshold=0.5, return_expert_info=False):
        all_probs = []
        all_expert_info = [] if return_expert_info else None
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if return_expert_info:
                    _, logits, expert_info = model(input_ids=input_ids, attention_mask=attention_mask, return_expert_info=True)
                    all_expert_info.append(expert_info)
                else:
                    _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(logits, dim=-1)[:, :, 1]
                all_probs.append(probs)

        stacked_probs = torch.stack(all_probs, dim=0)

        if self.method == 'soft_voting' or self.method == 'weighted':
            weights_tensor = torch.tensor(self.weights, device=stacked_probs.device).view(-1, 1, 1)
            avg_probs = (stacked_probs * weights_tensor).sum(dim=0)
            predictions = (avg_probs >= threshold).long()
        elif self.method == 'hard_voting':
            votes = (stacked_probs >= threshold).float()
            predictions = (votes.mean(dim=0) >= 0.5).long()
            avg_probs = stacked_probs.mean(dim=0)

        if return_expert_info:
            return predictions, avg_probs, all_expert_info
        return predictions, avg_probs

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self