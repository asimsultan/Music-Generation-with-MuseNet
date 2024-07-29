
import os
import torch
import argparse
from openai import MuseNet
from datasets import load_dataset
from utils import get_device, preprocess_midi, MusicDataset

def main(data_path):
    # Parameters
    model_name = 'musenet'
    batch_size = 8
    epochs = 3
    learning_rate = 5e-5

    # Load Dataset
    dataset = load_dataset('midi', data_files={'train': data_path})

    # Preprocess Data
    preprocessed_datasets = dataset.map(preprocess_midi, batched=True)

    # DataLoader
    train_dataset = MusicDataset(preprocessed_datasets['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    device = get_device()
    model = MuseNet.from_pretrained(model_name)
    model.to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / (num_training_steps * 0.1), 1.0)
    )

    # Training Function
    def train_epoch(model, data_loader, optimizer, device, scheduler):
        model.train()
        total_loss = 0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Training Loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')

    # Save Model
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save_pretrained(model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the MIDI files')
    args = parser.parse_args()
    main(args.data_path)
