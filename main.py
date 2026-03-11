from BERTMOE.dataset import prepare_data
from BERTMOE.config import CONFIG
from BERTMOE.inference import run_hybrid_pipeline
from BERTMOE.train import train_ensemble

# 1. Carrega os Dados
train_loader, val_loader, test_loader, tokenizer, train_data = prepare_data(CONFIG)

# 2. Roda o treinamento
ensemble_model = train_ensemble(train_loader, val_loader)

# 3. Roda o pipeline de dois estágios no conjunto de teste
results = run_hybrid_pipeline(test_loader.dataset.data, sklearn_model_path='ensemble_model.pkl')