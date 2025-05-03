"""
Скрипт для обучения модели определения депрессии.
Этот скрипт теперь является оберткой для модульной архитектуры в ml/.
"""

import os
import json
import argparse
from ml.preprocessing import preprocess_user
from ml.dataset import prepare_data, DepressionDataset
from ml.model import BiLSTMWithAttention, get_pos_weight
from ml.training import train_model, evaluate
from ml.balancing import balance_dataset
import torch
import fasttext
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Обучение модели для определения депрессии')
    
    parser.add_argument('--data_path', type=str, default='dataset/data.json',
                      help='Путь к файлу с данными')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Количество эпох обучения')
    parser.add_argument('--balance', action='store_true',
                      help='Использовать балансировку данных')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Загрузка данных
    print(f"Загрузка данных из {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Предобработка данных
    print("Предобработка данных...")
    users = [preprocess_user(user) for user in raw_data]
    texts = [user['text'] for user in users]
    labels = [user['label'] for user in users]
    meta_features = [list(user['meta'].values()) for user in users]
    
    # Вывод распределения классов
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"Распределение классов: Положительные - {pos_count}, Отрицательные - {neg_count}")
    
    # Подготовка данных
    texts_train, texts_val, metas_train, metas_val, labels_train, labels_val = prepare_data(
        texts, meta_features, labels
    )
    
    # Балансировка данных, если указан флаг
    if args.balance:
        print("Балансировка обучающих данных...")
        texts_train, metas_train, labels_train = balance_dataset(
            texts_train, metas_train, labels_train, method='random_oversample'
        )
        pos_count = sum(labels_train)
        neg_count = len(labels_train) - pos_count
        print(f"Новое распределение классов: Положительные - {pos_count}, Отрицательные - {neg_count}")
    
    # Загрузка модели FastText
    print("Загрузка модели FastText...")
    ft = fasttext.load_model("cc.ru.300.bin")
    
    # Создание датасетов
    print("Создание датасетов...")
    train_dataset = DepressionDataset(texts_train, metas_train, labels_train, ft_model=ft)
    val_dataset = DepressionDataset(texts_val, metas_val, labels_val, ft_model=ft)
    
    # Создание загрузчиков данных
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Выбор устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Создание модели
    embedding_dim = 300  # размерность FastText
    hidden_dim = 128
    meta_dim = 6  # количество мета признаков
    
    model = BiLSTMWithAttention(embedding_dim, hidden_dim, meta_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Расчет весов классов
    pos_weight, _ = get_pos_weight(labels)
    criterion = torch.nn.BCELoss()
    
    # Обучение модели
    print(f"Начало обучения на {args.epochs} эпохах...")
    model, history = train_model(
        model, train_loader, val_loader, optimizer, criterion, device, 
        num_epochs=args.epochs, model_name='depression_model.pt'
    )
    
    # Оценка модели
    print("Финальная оценка модели...")
    metrics = evaluate(model, val_loader, criterion, device)
    
    print(f"Финальные метрики:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    print("\nОбучение завершено!")
    print("Для использования модульной архитектуры используйте команды:")
    print("Обучение: python -m ml.train --help")
    print("Предсказание: python -m ml.predict --help")
    print("Скрипты доступны в папке ml/")

if __name__ == "__main__":
    main()