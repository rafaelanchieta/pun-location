import os
import joblib
import json
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import CONFIG, DEVICE
from .model import BertMoEClassifier, MoEEnsemble

def load_ensemble_from_saved_models(config, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    models = []

    for seed in config['ensemble_seeds']:
        model_path = os.path.join(config['output_dir'], f'moe_model_seed{seed}.pt')
        if not os.path.exists(model_path): 
            print(f"Aviso: Modelo {model_path} não encontrado.")
            continue

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


def main():
    print("=" * 60)
    print("PIPELINE HÍBRIDO: ENSEMBLE (SKLEARN) + BERT MOE")
    print("=" * 60)

    # 1. Carregar o modelo ensemble (sklearn)
    print("🔄 Carregando modelo ensemble (sklearn)...")
    try:
        # Usa o diretório onde o script está sendo executado. Pode ser necessário ajustar o path dependendo de onde fica o pkl.
        sklearn_path = 'ensemble_model.pkl' 
        if not os.path.exists(sklearn_path) and os.path.exists(f'../{sklearn_path}'):
            sklearn_path = f'../{sklearn_path}'

        ensemble_data = joblib.load(sklearn_path)
        sklearn_model = ensemble_data['model']
        vectorizer = ensemble_data['vectorizer']
        print("✅ Modelo ensemble carregado com sucesso!")
    except FileNotFoundError:
        print("❌ Erro: Arquivo 'ensemble_model.pkl' não encontrado.")
        raise

    # 2. Carregar os modelos BERT MoE
    print("🔄 Carregando modelos BERT MoE...")
    try:
        # Corrige caminhos relativos ao arquivo original de output caso execute deste diretório
        if not os.path.exists(CONFIG['output_dir']) and os.path.exists(f"../{CONFIG['output_dir']}"):
            CONFIG['output_dir'] = f"../{CONFIG['output_dir']}"

        ensemble, tokenizer = load_ensemble_from_saved_models(CONFIG, device=DEVICE)
        print("✅ Modelos BERT MoE carregados com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar BERT MoE: {e}")
        raise

    # 3. Carregar dataset de teste
    print("🔄 Carregando dataset de teste (Superar/Puntuguese)...")
    dataset = load_dataset("Superar/Puntuguese")
    test_data = dataset['test']
    print(f"✅ Dataset carregado: {len(test_data)} exemplos.")

    # 4. Executar pipeline híbrido
    print("\n🚀 Executando pipeline híbrido...")
    print("   1. Classificação da sentença (Ensemble Sklearn)")
    print("   2. Se positivo -> Localização de tokens (BERT MoE)")

    STAGE1_THRESHOLD = 0.4
    print(f"   ℹ️ Threshold do Estágio 1 ajustado para: {STAGE1_THRESHOLD}")

    y_true_tokens = []
    y_pred_tokens = []
    results = []

    total_sentences = 0
    detected_puns_stage1 = 0

    for example in tqdm(test_data, desc="Processando"):
        tokens = example['tokens']
        labels = example['labels']
        text = " ".join(tokens)
        
        # --- Estágio 1: Classificação da Sentença (Ensemble Sklearn) ---
        text_vectorized = vectorizer.transform([text])
        probs = sklearn_model.predict_proba(text_vectorized)[0]
        pun_prob = probs[1]
        
        sent_pred = 1 if pun_prob >= STAGE1_THRESHOLD else 0
        total_sentences += 1
        if sent_pred == 1:
            detected_puns_stage1 += 1
        
        pred_labels = [0] * len(tokens)
        token_probs = [0.0] * len(tokens) 
        
        if sent_pred == 1:
            # --- Estágio 2: Localização de Tokens (BERT MoE) ---
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=CONFIG['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            
            with torch.no_grad():
                bert_preds, bert_probs_tensor = ensemble.predict(input_ids, attention_mask)
                
            bert_preds = bert_preds[0].cpu().numpy()
            bert_probs = bert_probs_tensor[0].cpu().numpy()
            
            word_ids = encoding.word_ids(batch_index=0)
            previous_word_idx = None
            
            for idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != previous_word_idx:
                    pred_labels[word_idx] = int(bert_preds[idx])
                    token_probs[word_idx] = float(bert_probs[idx])
                previous_word_idx = word_idx
                
        y_true_tokens.extend(labels)
        y_pred_tokens.extend(pred_labels)
        
        results.append({
            'text': text,
            'tokens': tokens,
            'true_labels': labels,
            'pred_labels': pred_labels,
            'predicted_probs': [round(p, 4) for p in token_probs],
            'stage1_pred': int(sent_pred),
            'stage1_prob': float(pun_prob)
        })

    # 5. Relatório
    print("\n" + "="*60)
    print("📊 RESULTADOS DO PIPELINE HÍBRIDO")
    print("="*60)
    print(f"Sentenças processadas: {total_sentences}")
    print(f"Trocadilhos detectados (Estágio 1): {detected_puns_stage1} ({detected_puns_stage1/total_sentences:.1%})")

    print("\nRelatório de Classificação (Nível de Token):")
    print(classification_report(y_true_tokens, y_pred_tokens, target_names=['Não Pun', 'Pun'], digits=4))

    # Salvar resultados
    output_file = 'hybrid_predictions_prob.jsonl'
    
    # Salva na raiz do projeto original (um nível acima se estiver rodando dentro do diretorio BERTMOE)
    out_path = f"../{output_file}" if os.path.basename(os.getcwd()) == 'BERTMOE' else output_file
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
    print(f"\n💾 Resultados detalhados salvos em: {output_file}")
    print(f"   📊 Inclui 'predicted_probs' com probabilidades de cada token")

if __name__ == "__main__":
    main()