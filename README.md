# pun-location

Repositório para o treinamento e inferência de modelos voltados à localização de trocadilhos (puns) em português.

## Como executar

### 1. Preparar o ambiente
Recomenda-se o uso de um ambiente virtual (como `venv` ou `conda`) para isolar as dependências:

```bash
# Se utilizar venv
python -m venv .venv
source .venv/bin/activate  # (Linux/Mac)
# .venv\Scripts\activate   # (Windows)
```

### 2. Instalar as dependências
Instale as bibliotecas necessárias listadas em `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Executar a pipeline principal
O ponto de entrada do projeto é o script `main.py`, que encadeia a preparação de dados, o treinamento e as avaliações no pipeline híbrido:

```bash
python main.py
```