import torch
from tqdm import tqdm
import os

SEQUENCE_MODELS = ["rnn", "lstm", "transformer"]

def train(model, model_type, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, metrics_logger, results_path):

    print("\n")
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train", ascii=' >=')
        
        for _, (inputs, targets) in enumerate(pbar):
            
            optimizer.zero_grad()
            outputs = model(inputs)

            # If the model is a sequence model, get the last output
            # Don't worry about this till the RNN implementation onwards
            if model_type in SEQUENCE_MODELS:
                outputs = outputs[:,-1, :]
                
            # Compute the loss by calling the criterion
            loss = criterion(outputs, targets)
            # Backpropagate the loss
            loss.backward()
            # Update the model parameters using the optimizer's step
            optimizer.step()
            # Convert output logits to probabilities using softmax
            # Use torch.nn.functional.softmax and specify the correct dimension
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get the classification predictions from output probabilities
            # Hint: Use torch.argmax on the appropriate dimension
            predictions = torch.argmax(probabilities, dim=1)
            
            # Update metrics logger and progress bar
            num_correct = (predictions == targets).sum().item()
            metrics_logger.update('train_loss', loss.item())
            metrics_logger.update('train_accuracy', 100*num_correct/targets.size(0))
            pbar.set_postfix({
                'loss': f'{metrics_logger.get_epoch_average("train_loss"):.4f}',
                'accuracy': f'{metrics_logger.get_epoch_average("train_accuracy"):.2f}%'
            })
        

        pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation", ascii=' >=')
        for inputs, targets in pbar:
            with torch.no_grad():
                outputs = model(inputs)
                if model_type in SEQUENCE_MODELS:
                    outputs = outputs[:,-1, :]

                loss = criterion(outputs, targets)
                predictions = torch.argmax(outputs, dim=1)

                num_correct = (predictions == targets).sum().item()
                metrics_logger.update('val_loss', loss.item())
                metrics_logger.update('val_accuracy', 100*num_correct/targets.size(0))
                pbar.set_postfix({
                    'loss': f'{metrics_logger.get_epoch_average("val_loss"):.4f}',
                    'accuracy': f'{metrics_logger.get_epoch_average("val_accuracy"):.2f}%'
                })
        
        metrics_logger.next_epoch()
        print("\n")

    os.makedirs(os.path.join(results_path, "plots"), exist_ok=True)
    
    metrics_logger.plot_metric(["train_loss", "val_loss"], 
                             os.path.join(results_path, "plots", "train_loss.png"))
    metrics_logger.plot_metric(["train_accuracy", "val_accuracy"], 
                             os.path.join(results_path, "plots", "train_accuracy.png"))
    
    # Save model and results
    print(f"Saving train results to {os.path.join(results_path, 'train_results.txt')}") 
    with open(os.path.join(results_path, "train_results.txt"), "w") as f:
        f.write(f"Train Loss: {metrics_logger.get_last_epoch_average('train_loss'):.4f}\n")
        f.write(f"Train Accuracy: {metrics_logger.get_last_epoch_average('train_accuracy'):.2f}%\n")
        f.write(f"Validation Loss: {metrics_logger.get_last_epoch_average('val_loss'):.4f}\n")
        f.write(f"Validation Accuracy: {metrics_logger.get_last_epoch_average('val_accuracy'):.2f}%\n")

    torch.save(model, os.path.join(results_path, f'{model_type}.pth'))
    print(f"Model saved at {os.path.join(results_path, f'{model_type}.pth')}")

def test(model, model_type, test_dataloader, criterion, metrics_logger, results_path):

    print("\n")
    pbar = tqdm(test_dataloader, desc=f"Test", ascii=' >=')

    for _, (inputs, targets) in enumerate(pbar):
        with torch.no_grad():
            outputs = model(inputs)
            if model_type in SEQUENCE_MODELS:
                outputs = outputs[:,-1, :]

            loss = criterion(outputs, targets)

            predictions = torch.argmax(outputs, dim=1)

            num_correct = (predictions == targets).sum().item()
            metrics_logger.update('test_loss', loss.item())
            metrics_logger.update('test_accuracy', 100*num_correct/targets.size(0))

            pbar.set_postfix({'loss': f'{metrics_logger.get_epoch_average("test_loss"):.4f}', 
                                'accuracy': f'{metrics_logger.get_epoch_average("test_accuracy"):.2f}%'})
    print("\n")
    
    # Save results
    print(f"Saving test results to {os.path.join(results_path, 'test_results.txt')}")
    with open(os.path.join(results_path, 'test_results.txt'), "w") as f:
        f.write(f"Test Loss: {metrics_logger.get_epoch_average('test_loss'):.4f}\n")
        f.write(f"Test Accuracy: {metrics_logger.get_epoch_average('test_accuracy'):.2f}%\n")