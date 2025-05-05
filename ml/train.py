import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import fasttext
import argparse
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from ml.preprocessing import preprocess_user, preprocess_batch
from ml.dataset import DepressionDataset, prepare_data
from ml.model import BiLSTMWithAttention, get_pos_weight
from ml.training import train_model, evaluate
from ml.balancing import balance_dataset

def parse_args():
    """
    Парсинг аргументов командной строки
    """
    parser = argparse.ArgumentParser(description='Обучение модели для определения депрессии')
    
    parser.add_argument('--data_path', type=str, default='dataset/data.json',
                        help='Путь к файлу с данными')
    parser.add_argument('--fasttext_path', type=str, default='cc.ru.300.bin',
                        help='Путь к модели FastText')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Директория для сохранения моделей')
    parser.add_argument('--model_name', type=str, default='depression_model.pt',
                        help='Имя файла модели')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Размерность скрытого состояния LSTM')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Вероятность дропаута')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Скорость обучения')
    parser.add_argument('--patience', type=int, default=5,
                        help='Количество эпох без улучшения до остановки')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Доля данных для валидации')
    parser.add_argument('--balance_method', type=str, choices=['none', 'random_oversample', 'random_undersample', 'smote'],
                        default='random_oversample', help='Метод балансировки данных')
    parser.add_argument('--max_len', type=int, default=500,
                        help='Максимальная длина последовательности')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Использовать планировщик скорости обучения')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Директория для результатов')
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help='Количество слоёв LSTM')
    parser.add_argument('--handle_outliers', action='store_true',
                        help='Обрабатывать выбросы в метаданных')
    parser.add_argument('--augment_positive', action='store_true',
                        help='Аугментировать положительные примеры')
    
    return parser.parse_args()

def plot_metrics(history: List[Dict[str, float]], output_dir: str):
    """
    Построение графиков метрик
    
    Args:
        history: история обучения
        output_dir: директория для сохранения графиков
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Построение графика потерь
    plt.figure(figsize=(10, 6))
    plt.plot([x['epoch'] for x in history], [x['train_loss'] for x in history], label='Train Loss')
    plt.plot([x['epoch'] for x in history], [x['loss'] for x in history], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    
    # Построение графика точности
    plt.figure(figsize=(10, 6))
    plt.plot([x['epoch'] for x in history], [x['accuracy'] for x in history], label='Accuracy')
    plt.plot([x['epoch'] for x in history], [x['precision'] for x in history], label='Precision')
    plt.plot([x['epoch'] for x in history], [x['recall'] for x in history], label='Recall')
    plt.plot([x['epoch'] for x in history], [x['f1'] for x in history], label='F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training Metrics')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'metrics.png'))

def main():
    args = parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    print(f"Загрузка данных из {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print("Предобработка данных...")
    if args.handle_outliers or args.augment_positive:
        users, stats = preprocess_batch(
            raw_data, 
            augment_positive=args.augment_positive,
            handle_meta_outliers=args.handle_outliers
        )
        print(f"Статистика предобработки:")
        print(f"  Исходные примеры: {stats['original_count']}")
        print(f"  Позитивные примеры: {stats['positive_count']}")
        print(f"  Негативные примеры: {stats['negative_count']}")
        print(f"  Добавлено аугментированных: {stats['augmented_count']}")
    else:
        users = [preprocess_user(user) for user in raw_data]
    
    texts = [user['text'] for user in users]
    labels = [user['label'] for user in users]
    meta_features = [list(user['meta'].values()) for user in users]
    
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"Распределение классов: Позитивные - {pos_count}, Негативные - {neg_count}")
    
    print(f"Загрузка модели FastText из {args.fasttext_path}...")
    ft = fasttext.load_model(args.fasttext_path)
    
    # Стандартное обучение без K-fold
    # Подготовка данных
    texts_train, texts_val, metas_train, metas_val, labels_train, labels_val = prepare_data(
        texts, meta_features, labels, test_size=args.test_size
    )
    
    # Балансировка обучающих данных
    if args.balance_method != 'none':
        print(f"Балансировка данных с использованием метода: {args.balance_method}...")
        texts_train, metas_train, labels_train = balance_dataset(
            texts_train, metas_train, labels_train, method=args.balance_method
        )
        print(f"После балансировки: {len(texts_train)} примеров")
        pos_count = sum(labels_train)
        neg_count = len(labels_train) - pos_count
        print(f"Новое распределение классов: Позитивные - {pos_count}, Негативные - {neg_count}")
    
    # Создание датасетов
    print("Создание датасетов...")
    train_dataset = DepressionDataset(
        texts_train, metas_train, labels_train, ft_model=ft, max_len=args.max_len
    )
    val_dataset = DepressionDataset(
        texts_val, metas_val, labels_val, ft_model=ft, max_len=args.max_len
    )
    
    # Создание загрузчиков данных
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Создание модели
    print("Создание модели...")
    embedding_dim = 300  # размерность FastText
    meta_dim = len(meta_features[0]) if meta_features else 6
    
    model = BiLSTMWithAttention(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        meta_dim=meta_dim,
        dropout=args.dropout,
        lstm_layers=args.lstm_layers
    ).to(device)
    
    # Оптимизатор
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Веса классов
    pos_weight, _ = get_pos_weight(labels)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Обучение модели
    print("Начало обучения...")
    start_time = datetime.now()
    
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        model_save_path=args.model_dir,
        model_name=args.model_name
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Обучение завершено за {training_time}")
    
    # Оценка финальной модели
    print("Оценка модели на валидационных данных...")
    val_metrics = evaluate(model, val_loader, criterion, device)
    
    print(f"Валидационные метрики:")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")
    print(f"  AUC: {val_metrics['auc']:.4f}")
    
    # Построение графиков
    print("Построение графиков метрик...")
    plot_metrics(history, args.output_dir)
    
    # Сохранение метрик
    results = {
        'args': vars(args),
        'final_metrics': val_metrics,
        'history': history,
        'training_time': str(training_time)
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Результаты сохранены в {os.path.join(args.output_dir, 'results.json')}")
    print(f"Модель сохранена в {os.path.join(args.model_dir, args.model_name)}")

if __name__ == "__main__":
    main() 