import os
import logging
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.models.bert import BertForSequenceClassification, BertTokenizer
from transformers.trainer_callback import EarlyStoppingCallback
from sklearn.metrics import f1_score, accuracy_score


class MyTrainer():
    def __init__(self, model, tokenizer, train_dataset, dev_dataset, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

    def set_logger(self, output_dir, name):
        log_file_path = os.path.join(output_dir, "training.log")
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                               datefmt='%m/%d/%Y %H:%M:%S')
            file_handler.setFormatter(file_formatter)

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_formatter = logging.Formatter('%(message)s')
            stream_handler.setFormatter(stream_formatter)

            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
        return logger

    def compute_metrics(self, p):
        preds = p.predictions.argmax(-1)
        return {
            'accuracy': accuracy_score(p.label_ids, preds),
            'f1': f1_score(p.label_ids, preds, average='weighted')
        }

    def training(self, args, device):
        print('--------------- begin training ---------------')
        logger = self.set_logger(args.logging_dir, 'my_logger')
        logger.info("dataset: {}".format(args.data_dir))
        logger.info("model: {}".format(args.model_name))

        if self.dev_dataset is not None:
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                evaluation_strategy="epoch",  # "no", "steps", "epoch"
                learning_rate=args.learning_rate,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.train_bsize,
                save_strategy="epoch",
                load_best_model_at_end=True,
                # metric_for_best_model="eval_accuracy",
                weight_decay=0,
                logging_dir=args.logging_dir,
                logging_strategy="epoch",
                seed=args.seed,
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.dev_dataset,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
            )
        else:
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                evaluation_strategy="no",  # "no", "steps", "epoch"
                learning_rate=args.learning_rate,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.train_bsize,
                save_strategy="no",
                weight_decay=0,
                logging_dir=args.logging_dir,
                logging_strategy="epoch",
                seed=args.seed,
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                compute_metrics=self.compute_metrics,
            )

        trainer.train()
        # trainer.evaluate()

        if args.save_model:
            trainer.save_model(args.output_dir)

        for obj in trainer.state.log_history:
            logger.info(str(obj))
        '''
        train_loader = DataLoader(train_dataset, batch_size=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        model.train()
        for epoch in range(args.max_epoch):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
        '''

    def testing(self, args, device):
        print('--------------- begin testing! ---------------')
        dataloader = DataLoader(self.test_dataset, batch_size=args.test_bsize)
        # model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
        model = self.model

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, 1)
                predicted = predicted.cpu().numpy()
                labels = labels.cpu().numpy()
        acc = accuracy_score(labels, predicted)
        f1 = f1_score(labels, predicted, average='weighted')
        print(f"Accuracy: {acc}, F1 Score: {f1}")

    def train(self, args, device):
        if args.train:
            self.training(args, device)
        if args.test:
            self.testing(args, device)
        if not args.train and not args.test:
            print("No action specified. Please use --train and/or --test.")
