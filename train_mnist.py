import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from psthree.cnn import CNN, ManualCNN
from psthree.data import load_mnist_dataset
from psthree.utils import seed_everything, get_results_path
from psthree.utils import MetricsLogger
from psthree.train_test import train, test
import argparse

def train_cnn(model_type, train_dataset, batch_size, num_epochs, learning_rate, results_path, train_val_split=0.8):
    train_data = Subset(train_dataset, indices=range(int(len(train_dataset) * train_val_split)))
    val_data = Subset(train_dataset, indices=range(int(len(train_dataset) * train_val_split), len(train_dataset)))

    if model_type == "manual_cnn":
        model = ManualCNN()
    elif model_type == "cnn":
        model = CNN()
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    metrics_logger = MetricsLogger()

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    train(model, model_type, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, metrics_logger, results_path)

    return model

def test_cnn(model, model_type, test_dataset, batch_size, results_path):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    metrics_logger = MetricsLogger()
    
    test(model, model_type, test_dataloader, criterion, metrics_logger, results_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "manual_cnn"])
    parser.add_argument("--results_path", type=str, default="results")
    args = parser.parse_args()

    seed_everything()
    results_path = get_results_path(args.results_path, args.model_type, "mnist")
    train_dataset, test_dataset = load_mnist_dataset()

    model = train_cnn(args.model_type, train_dataset, args.batch_size, args.num_epochs, args.learning_rate, results_path)

    test_cnn(model, args.model_type, test_dataset, args.batch_size, results_path)