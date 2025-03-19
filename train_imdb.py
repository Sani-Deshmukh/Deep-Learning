import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from psthree.rnn import RNN
from psthree.lstm import LSTM
from psthree.transformer import Transformer
from psthree.data import IMDBDataset
from psthree.train_test import train, test
from psthree.utils import MetricsLogger, seed_everything, get_results_path
import argparse

def train_imdb(model_type, train_dataset, num_epochs, batch_size, learning_rate, results_path, train_val_split=0.8):
    train_data = train_dataset.data[:int(len(train_dataset.data) * train_val_split)]
    val_data = train_dataset.data[int(len(train_dataset.data) * train_val_split):]

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    if model_type == 'rnn':
        model = RNN(train_dataset.tokenizer, embedding_size=128, hidden_size=256, output_size=2)
    elif model_type == 'lstm':
        model = LSTM(train_dataset.tokenizer, embedding_size=128, hidden_size=108, output_size=2)
    elif model_type == 'transformer':
        model = Transformer(train_dataset.tokenizer, embedding_size=128, num_heads=8, d_ff=128, output_size=2)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    metrics_logger = MetricsLogger()

    train(model, model_type, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, metrics_logger, results_path)

    return model

def test_imdb(model, model_type, test_dataset, batch_size, results_path):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    metrics_logger = MetricsLogger()
    
    test(model, model_type, test_dataloader, criterion, metrics_logger, results_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dataset_length", type=int, default=-1)
    parser.add_argument("--min_seq_length", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=200)
    parser.add_argument("--model_type", type=str, default="rnn", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--results_path", type=str, default="results")
    args = parser.parse_args()

    seed_everything()
    results_path = get_results_path(args.results_path, args.model_type, "imdb")
    
    train_dataset = IMDBDataset(split='train', min_seq_length=args.min_seq_length, max_seq_length=args.max_seq_length, dataset_length=args.dataset_length)
    test_dataset = IMDBDataset(split='test', tokenizer=train_dataset.tokenizer, min_seq_length=args.min_seq_length, max_seq_length=args.max_seq_length, dataset_length=args.dataset_length)
    
    model = train_imdb(args.model_type, train_dataset, args.num_epochs, args.batch_size, args.learning_rate, results_path)
    
    test_imdb(model, args.model_type, test_dataset, args.batch_size, results_path)