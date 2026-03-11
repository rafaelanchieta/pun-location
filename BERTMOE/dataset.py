import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

def count_class_distribution(dataset):
    counts = {0: 0, 1: 0}
    for example in dataset:
        labels = example['labels']
        for label in labels:
            if label != -100 and label in counts:
                counts[label] += 1
    return counts

class PunDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, label2id=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or {'O': 0, 'PUN': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        labels = item['labels']

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx])
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

def prepare_data(config):
    dataset = load_dataset(config['dataset_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    train_data = list(dataset['train'])
    val_data = list(dataset['validation'])
    test_data = list(dataset['test'])

    train_dataset = PunDataset(train_data, tokenizer, config['max_length'])
    val_dataset = PunDataset(val_data, tokenizer, config['max_length'])
    test_dataset = PunDataset(test_data, tokenizer, config['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    return train_loader, val_loader, test_loader, tokenizer, train_data