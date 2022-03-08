import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM

import data_prep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")


def train(dataset, model):
    # stage data for training
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # set model to train
    model.to(device)
    model.train()

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    epochs = 2
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()

            # prep data for predict step
            masked_inputs = data_prep.masking_step(batch["input_ids"]).to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["mlm_labels"].to(device)

            outputs = model(
                masked_inputs,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch} Loss: {loss.item()}")
        model.save_pretrained(f"models/URLTran-BERT-{epoch}")


def predict_mask(url, tokenizer, model):
    inputs = data_prep.preprocess(url, tokenizer)
    masked_inputs = data_prep.masking_step(inputs["input_ids"]).to(device)
    with torch.no_grad():
        predictions = model(masked_inputs)

    output_ids = torch.argmax(
        torch.nn.functional.softmax(predictions.logits[0], -1), dim=1).tolist()

    return masked_inputs, output_ids


if __name__ == "__main__":
    data_path = "data/final_data.csv"
    dataset = data_prep.URLTranDataset(data_path, tokenizer)
    train(dataset, model)

    # Example Inference
    url = "huggingface.co/docs/transformers/task_summary"
    input_ids, output_ids = predict_mask(url, tokenizer, model)
    masked_input = tokenizer.decode(input_ids[0].tolist()).replace(' ', '')
    prediction = tokenizer.decode(output_ids).replace(' ', '')

    print(f"Masked Input: {masked_input}")
    print(f"Predicted Output: {prediction}")
