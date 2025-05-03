import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from typing import Dict, Any, Tuple, List, Optional, Callable
import os
from tqdm import tqdm
from copy import deepcopy
import time

class EarlyStopping:
    """
    Ранняя остановка обучения при отсутствии улучшений
    """
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
        else:  # mode == 'max'
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
    Обучение в течение одной эпохи
    
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

            probs = model(text, meta)
            
            # Используем BCE Loss
            loss = criterion(probs, labels)
            total_loss += loss.item()
            
            preds = (probs > threshold).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Вычисление метрик
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    
    # ROC-AUC
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
    # Создаем директорию для сохранения модели, если еще не существует
    os.makedirs(model_save_path, exist_ok=True)
    
    # Инициализация EarlyStopping
    early_stopping = EarlyStopping(patience=patience, mode='max')
    
    # Инициализация scheduler для изменения learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    history = []
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        # Обучение
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Оценка
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Запись метрик
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **val_metrics
        }
        history.append(metrics)
        
        # Вывод метрик
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Сохранение лучшей модели
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
            print(f"  Saved best model with F1: {best_f1:.4f}")
        
        # Обновление learning rate
        scheduler.step(val_metrics['f1'])
        
        # Проверка ранней остановки
        if early_stopping(val_metrics['f1']):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load(os.path.join(model_save_path, model_name)))
    
    return model, history

def train_with_kfold(dataset: Dataset,
                    model_factory: Callable[[], nn.Module],
                    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
                    criterion: nn.Module,
                    device: torch.device,
                    n_splits: int = 5,
                    num_epochs: int = 10,
                    patience: int = 5,
                    batch_size: int = 32,
                    model_save_path: str = 'models',
                    base_model_name: str = 'depression_model') -> Tuple[nn.Module, List[Dict[str, float]]]:
    """
    Обучение модели с использованием K-fold перекрестной валидации
    
    Args:
        dataset: полный набор данных
        model_factory: функция для создания модели
        optimizer_factory: функция для создания оптимизатора
        criterion: функция потерь
        device: устройство
        n_splits: количество разбиений (складок)
        num_epochs: количество эпох
        patience: количество эпох без улучшения до остановки
        batch_size: размер батча
        model_save_path: путь для сохранения моделей
        base_model_name: базовое имя файлов моделей
        
    Returns:
        Tuple: финальная модель и история обучения
    """
    # Создаем директорию для сохранения модели, если еще не существует
    os.makedirs(model_save_path, exist_ok=True)
    
    # Создаем разбиения
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Инициализируем метрики для каждой складки
    fold_metrics = []
    fold_models = []
    
    # Получаем все индексы
    indices = np.arange(len(dataset))
    
    # Обучаем модель на каждой складке
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n{'='*20} Fold {fold+1}/{n_splits} {'='*20}")
        
        # Создаем загрузчики данных для текущей складки
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False
        )
        
        # Создаем модель и оптимизатор
        model = model_factory().to(device)
        optimizer = optimizer_factory(model)
        
        # Обучаем модель
        model_name = f"{base_model_name}_fold{fold+1}.pt"
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=num_epochs,
            patience=patience,
            model_save_path=model_save_path,
            model_name=model_name
        )
        
        # Сохраняем метрики и модель
        final_metrics = history[-1]
        fold_metrics.append(final_metrics)
        fold_models.append(model.state_dict())
        
        print(f"\nFold {fold+1} Results:")
        print(f"  F1: {final_metrics['f1']:.4f}")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall: {final_metrics['recall']:.4f}")
        print(f"  AUC: {final_metrics['auc']:.4f}")
    
    # Вычисляем средние метрики
    avg_metrics = {}
    for metric in ['f1', 'accuracy', 'precision', 'recall', 'auc', 'loss']:
        avg_metrics[metric] = np.mean([m[metric] for m in fold_metrics])
    
    # Выводим средние метрики
    print("\n" + "="*50)
    print("Average Metrics Across All Folds:")
    print(f"  F1: {avg_metrics['f1']:.4f}")
    print(f"  Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall: {avg_metrics['recall']:.4f}")
    print(f"  AUC: {avg_metrics['auc']:.4f}")
    print("="*50)
    
    # Создаем финальную модель
    final_model = model_factory().to(device)
    
    # Загружаем лучшую модель (с самым высоким F1)
    best_fold_idx = np.argmax([m['f1'] for m in fold_metrics])
    best_model_path = os.path.join(model_save_path, f"{base_model_name}_fold{best_fold_idx+1}.pt")
    
    print(f"\nLoading best model from fold {best_fold_idx+1}")
    final_model.load_state_dict(torch.load(best_model_path))
    
    # Сохраняем ансамбль моделей
    ensemble_path = os.path.join(model_save_path, f"{base_model_name}_ensemble.pt")
    torch.save({
        'fold_models': fold_models,
        'best_model': final_model.state_dict(),
        'fold_metrics': fold_metrics,
        'avg_metrics': avg_metrics
    }, ensemble_path)
    
    print(f"Saved ensemble model to {ensemble_path}")
    
    # Возвращаем финальную модель и метрики
    return final_model, fold_metrics 