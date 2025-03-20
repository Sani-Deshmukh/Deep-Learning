import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import random
import os
from datetime import datetime

SEED = 42

class MetricsLogger:
    def __init__(self):
        self.step_metrics = defaultdict(list)
        self.epoch_metrics = [defaultdict(list)]
        self.current_epoch = 0
        
        # Initialize metrics we want to track
        self.tracked_metrics = [
            'train_loss', 'train_accuracy',
            'val_loss', 'val_accuracy', 
            'test_loss', 'test_accuracy'
        ]
        
    def update(self, metric_name, value):
        if metric_name in self.tracked_metrics:
            self.epoch_metrics[self.current_epoch][metric_name].append(value)
            self.step_metrics[f"{metric_name}_step"].append(value)
                
    def next_epoch(self):
        self.current_epoch += 1
        self.epoch_metrics.append(defaultdict(list))
        
    def get_epoch_average(self, metric_name, epoch=None):
        if metric_name not in self.tracked_metrics:
            return None
        if epoch is None:
            epoch = self.current_epoch
        return np.mean(self.epoch_metrics[epoch][metric_name])
    
    def get_last_epoch_average(self, metric_name):
        return self.get_epoch_average(metric_name, epoch=self.current_epoch-1)

    
    def plot_metric(self, metric_names, save_path, plot_epochs=True):
        if isinstance(metric_names, str):
            metric_names = [metric_names]

        metric_type = metric_names[0].split('_')[1]
            
        if plot_epochs:
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            for metric_name in metric_names:
                epoch_averages = []
                for epoch in range(len(self.epoch_metrics)):
                    if len(self.epoch_metrics[epoch][metric_name]) > 0:
                        avg = np.mean(self.epoch_metrics[epoch][metric_name])
                        epoch_averages.append(avg)
                
                ax1.plot(range(len(epoch_averages)), epoch_averages, label=metric_name)
                
                if len(self.step_metrics[f"{metric_name}_step"]) > 0 and 'val' not in metric_name:
                    values = self.step_metrics[f"{metric_name}_step"]
                    ax2.plot(range(len(values)), values, label=metric_name)
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel(metric_type.capitalize())
            ax1.set_title(f'{metric_type.capitalize()} Average vs Epoch')
            ax1.grid(True)
            ax1.legend()
            
            ax2.set_xlabel('Step')
            ax2.set_ylabel(metric_type.capitalize())
            ax2.set_title(f'{metric_type.capitalize()} vs Step')
            ax2.grid(True)
            ax2.legend()
            
        else:
            plt.figure(figsize=(10, 4))
            for metric_name in metric_names:
                values = self.step_metrics[f"{metric_name}_step"]
                plt.plot(range(len(values)), values, label=metric_name)
                
            plt.xlabel('Step')
            plt.ylabel(metric_type.capitalize())
            plt.title(f'Model {metric_type.capitalize()} Over Steps')
            plt.grid(True)
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(save_path)

def seed_everything():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(SEED)

def get_results_path(results_path, model_type, dataset_name=None    ):
    timestamp = datetime.now().strftime("%H-%M-%S_%m-%d-%Y")
    if dataset_name is not None:
        results_path = os.path.join(results_path, model_type, dataset_name, timestamp)
    else:
        results_path = os.path.join(results_path, model_type, timestamp)
    os.makedirs(results_path, exist_ok=True)
    return results_path