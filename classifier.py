import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, BertTokenizer

import data_prep

model_ckpt = "URLTran-BERT"
config = AutoConfig.from_pretrained(model_ckpt)
config.num_labels = 2
config.problem_type = "single_label_classification"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(url, tokenizer, model):
    inputs = data_prep.preprocess(url, tokenizer)
    return torch.argmax(torch.softmax(model(**inputs).logits, dim=1)).tolist()


def train_model(train_dataset, model):
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # model training
    model.to(device)
    model.train()

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            # prep data for predict step
            inputs = batch["input_ids"]
            labels = batch["label"]
            X = inputs.to("cpu")
            y = labels.to("cpu")

            outputs = model(X, labels=y)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch} Loss: {loss.item()}")
        model.save_pretrained(f"models/URLTran-BERT-CLS-{epoch}")


def eval_model(eval_dataset, tokenizer, model):
    eval_loader = DataLoader(eval_dataset, batch_size=2000, shuffle=True)

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            inputs = batch["input_ids"]
            labels = batch["label"]
            X_eval = inputs.to("cpu")
            y_eval = labels.to("cpu")

            outputs = model(X_eval, labels=y_eval)
            predictions = [
                torch.argmax(pred).tolist()
                for pred in torch.softmax(outputs.logits, dim=1)
            ]

            y_eval_true = y_eval.tolist()

            y_true.extend(y_eval_true)
            y_pred.extend(predictions)

        total_acc = accuracy_score(y_true, y_pred)
        total_f1 = f1_score(y_true, y_pred)
        print(f"Acc: {total_acc} F1: {total_f1}")


if __name__ == "__main__":
    data_path = "data/final_data.csv"
    dataset = data_prep.URLTranDataset(data_path, tokenizer)
    train_model(dataset, model)
