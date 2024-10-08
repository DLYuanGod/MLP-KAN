import os
import pdb
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, mode, data_dir, tokenizer, max_length=64):
        if mode == 'train':
            file_path = os.path.join(data_dir, 'train.txt')
        elif mode == 'dev':
            file_path = os.path.join(data_dir, 'dev.txt')
        elif mode == 'test':
            file_path = os.path.join(data_dir, 'test.txt')
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            datas = file.readlines()

        if "trec6" in data_dir:
            self.texts = [sample.split(":")[1].split(" ", 1)[1].strip() for sample in datas]
            labels = [sample.split(":")[0] for sample in datas]
        elif "sst2" in data_dir:
            self.texts = [sample.split(maxsplit=1)[1].strip() for sample in datas]
            labels = [int(sample.split(maxsplit=1)[0]) for sample in datas]
        self.labels, _ = self.encode_labels(labels)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode_labels(self, labels):
        label2id = {label: idx for idx, label in enumerate(set(labels))}
        encoded_labels = [label2id[label] for label in labels]
        return encoded_labels, label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
