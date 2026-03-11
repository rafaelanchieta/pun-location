import os
import random
import numpy as np
import torch

# Reprodutibilidade
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Verifica GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configurações Globais (v8 - Ensemble de Modelos)
CONFIG = {
    'model_name': 'neuralmind/bert-base-portuguese-cased',
    'max_length': 128,
    'batch_size': 8,
    'epochs': 10,
    'learning_rate': 1e-5,
    'warmup_steps': 200,
    'weight_decay': 0.05,
    'scheduler_type': 'linear',
    'num_experts': 4,
    'top_k': 2,
    'expert_hidden_dim': 256,
    'load_balance_loss_coef': 0.1,
    'dropout': 0.35,
    'label_smoothing': 0.1,
    'early_stopping_patience': 2,
    'num_labels': 2,
    'dataset_name': 'Superar/Puntuguese',
    'output_dir': './bert_moe_output',

    # Focal Loss 
    'use_focal_loss': True,
    'focal_alpha': 0.75,
    'focal_gamma': 2.0,

    # Layer Concatenation
    'concat_last_n_layers': 4,

    # Outras Features 
    'use_crf': False,
    'gradual_unfreezing': False,

    # ENSEMBLE
    'ensemble_size': 3,
    'ensemble_seeds': [42, 123, 456],
    'ensemble_method': 'soft_voting', 
}

if not os.path.exists(CONFIG['output_dir']):
    os.makedirs(CONFIG['output_dir'])