import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from typing import Dict, Any, Tuple, List, Optional, Callable
import os
from tqdm import tqdm
from copy import deepcopy
import time

class EarlyStopping:

    def __init__(self, patience: int = 5, min_delta: float = 0, mode: str = 'min'):
        """
        Инициализация
        
        Args:
            patience: количество эпох без улучшения до остановки
            min_delta: минимальное изменение для считания улучшением
            mode: 'min' для минимизации метрики, 'max' для максимизации
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
    
    def __call__(self, score: float) -> bool:
        """
        Проверка необходимости ранней остановки
        
        Args:
            score: текущее значение метрики
            
        Returns:
            bool: True, если нужно остановиться, иначе False
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, 
                device: torch.device) -> float:
    """
    Обучение модели на одной эпохе
    
    Args:
        model: модель
        dataloader: загрузчик данных
        optimizer: оптимизатор
        criterion: функция потерь
        device: устройство (CPU/GPU)
        
    Returns:
        float: средняя потеря за эпоху
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        text = batch['text'].to(device)
        meta = batch['meta'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(text, meta)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model: nn.Module, 
             dataloader: DataLoader, 
             criterion: nn.Module, 
             device: torch.device,
             threshold: float = 0.5) -> Dict[str, float]:
    """
    Оценка модели
    
    Args:
        model: модель
        dataloader: загрузчик данных
        criterion: функция потерь
        device: устройство (CPU/GPU)
        threshold: порог для бинарной классификации
        
    Returns:
        Dict: словарь с метриками
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text = batch['text'].to(device)
            meta = batch['meta'].to(device)
            labels = batch['label'].to(device)

            logits = model(text, meta)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, 
                device: torch.device,
                num_epochs: int = 10,
                patience: int = 5,
                model_save_path: str = 'models',
                model_name: str = 'depression_model.pt') -> Tuple[nn.Module, List[Dict[str, float]]]:
    """
    Полный цикл обучения модели
    
    Args:
        model: модель
        train_loader: загрузчик обучающих данных
        val_loader: загрузчик валидационных данных
        optimizer: оптимизатор
        criterion: функция потерь
        device: устройство
        num_epochs: количество эпох
        patience: количество эпох без улучшения до остановки
        model_save_path: путь для сохранения модели
        model_name: имя файла модели
        
    Returns:
        Tuple: обученная модель и история обучения
    """
    os.makedirs(model_save_path, exist_ok=True)
    
    early_stopping = EarlyStopping(patience=patience, mode='max')
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    history = []
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **val_metrics
        }
        history.append(metrics)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
            print(f"  Saved best model with F1: {best_f1:.4f}")
        
        scheduler.step(val_metrics['f1'])
        
        if early_stopping(val_metrics['f1']):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load(os.path.join(model_save_path, model_name)))
    
    return model, history 