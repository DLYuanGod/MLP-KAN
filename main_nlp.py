import os
import sys
import pdb
import random
import argparse
import numpy as np
import torch

from transformers.models.bert import BertForSequenceClassification, BertTokenizer
from dataset import MyDataset
from trainer import MyTrainer


def setup_seed(seed=3407):
    os.environ['PYTHONASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/sst2', type=str, help="dataset dir")
    parser.add_argument("--model_name", default='roberta-base', type=str)
    parser.add_argument("--tokenizer_name", default='roberta-base', type=str)
    parser.add_argument("--output_dir", default='./logs/runs', type=str, help="dir to save finetuned model")
    parser.add_argument("--logging_dir", default="./logs", type=str, help="dir to save logs")
    parser.add_argument('--save_model', action='store_true', help='saving the model')
    parser.add_argument("--train", action='store_true', help="include to perform training")
    parser.add_argument("--test", action='store_true', help="include to perform testing")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--epochs", default=5, type=int, help="total number of epoch")
    parser.add_argument("--train_bsize", default=128, type=int, help="training batch size")
    parser.add_argument("--test_bsize", default=256, type=int, help="evaluation batch size")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--patience", default=10, type=int, help="early stopping patience")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)    # return loss, logits
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    train_dataset = MyDataset("train", args.data_dir, tokenizer, max_length=64)
    test_dataset = MyDataset("test", args.data_dir, tokenizer, max_length=64)
    dev_dataset = None
    if "sst2" in args.data_dir:
        dev_dataset = MyDataset("dev", args.data_dir, tokenizer, max_length=64)

    trainer = MyTrainer(model, tokenizer, train_dataset, dev_dataset, test_dataset)
    trainer.train(args, device)
