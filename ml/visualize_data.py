import json
import os
import argparse
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml.visualization import (
    plot_class_distribution, 
    plot_meta_feature_distributions
)
from ml.preprocessing import preprocess_user

def parse_args():
    """
    Парсинг аргументов командной строки
    """
    parser = argparse.ArgumentParser(description='Визуализация данных')
    
    parser.add_argument('--data_path', type=str, default='dataset/data.json',
                      help='Путь к файлу с данными')
    parser.add_argument('--output_dir', type=str, default='visualizations/data',
                      help='Директория для сохранения визуализаций')
    
    return parser.parse_args()

def visualize_text_lengths(texts: List[str], labels: List[int], output_dir: str):
    """
    Визуализация длин текстов
    
    Args:
        texts: список текстов
        labels: список меток
        output_dir: директория для сохранения графиков
    """
    text_lengths = [len(text.split()) for text in texts]
    
    df = pd.DataFrame({
        'length': text_lengths,
        'label': labels
    })
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='length', hue='label', multiple='dodge', 
               bins=50, palette={0: '#4CAF50', 1: '#F44336'})
    plt.title('Распределение длин текстов')
    plt.xlabel('Длина текста (количество слов)')
    plt.ylabel('Количество примеров')
    plt.legend(['Нет депрессии', 'Есть депрессия'])
    
    plt.xlim(0, np.percentile(text_lengths, 99))  # отсекаем верхний 1% для лучшей видимости
    
    plt.savefig(os.path.join(output_dir, 'text_lengths_distribution.png'), bbox_inches='tight')
    plt.close()
    
    stats = {
        'mean': np.mean(text_lengths),
        'median': np.median(text_lengths),
        'min': np.min(text_lengths),
        'max': np.max(text_lengths),
        'std': np.std(text_lengths),
        'percentile_25': np.percentile(text_lengths, 25),
        'percentile_75': np.percentile(text_lengths, 75),
        'percentile_95': np.percentile(text_lengths, 95)
    }
    
    with open(os.path.join(output_dir, 'text_lengths_stats.txt'), 'w') as f:
        f.write("Статистика по длинам текстов (в словах):\n")
        for key, value in stats.items():
            f.write(f"  {key}: {value:.2f}\n")
    
    for label in [0, 1]:
        label_lengths = [length for length, l in zip(text_lengths, labels) if l == label]
        
        label_stats = {
            'mean': np.mean(label_lengths),
            'median': np.median(label_lengths),
            'min': np.min(label_lengths),
            'max': np.max(label_lengths),
            'std': np.std(label_lengths),
            'percentile_25': np.percentile(label_lengths, 25),
            'percentile_75': np.percentile(label_lengths, 75),
            'percentile_95': np.percentile(label_lengths, 95)
        }
        
        with open(os.path.join(output_dir, f'text_lengths_stats_class_{label}.txt'), 'w') as f:
            f.write(f"Статистика по длинам текстов для класса {label} (в словах):\n")
            for key, value in label_stats.items():
                f.write(f"  {key}: {value:.2f}\n")

def visualize_meta_features_correlation(meta_features: List[List[float]], 
                                       feature_names: List[str],
                                       output_dir: str):
    """
    Визуализация корреляции между мета-признаками
    
    Args:
        meta_features: список мета-признаков
        feature_names: имена признаков
        output_dir: директория для сохранения графиков
    """
    df = pd.DataFrame(meta_features, columns=feature_names)
    
    corr = df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Корреляция между мета-признаками')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'meta_features_correlation.png'), bbox_inches='tight')
    plt.close()

def main():
    """
    Основная функция для визуализации данных
    """
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Загрузка данных из {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print("Предобработка данных...")
    users = [preprocess_user(user) for user in raw_data]
    texts = [user['text'] for user in users]
    labels = [user['label'] for user in users]
    meta_features = [list(user['meta'].values()) for user in users]
    
    feature_names = [
        'sex', 'followers_count', 'alcohol', 
        'smoking', 'life_main', 'people_main'
    ]
    
    print("Визуализация распределения классов...")
    plot_class_distribution(
        labels, 
        title="Распределение классов в датасете", 
        save_path=os.path.join(args.output_dir, 'class_distribution.png')
    )
    
    print("Визуализация длин текстов...")
    visualize_text_lengths(texts, labels, args.output_dir)
    
    print("Визуализация распределения мета-признаков...")
    plot_meta_feature_distributions(
        meta_features, 
        labels, 
        feature_names, 
        save_dir=args.output_dir
    )
    
    print("Визуализация корреляции между мета-признаками...")
    visualize_meta_features_correlation(meta_features, feature_names, args.output_dir)
    
    print(f"Визуализации сохранены в {args.output_dir}")

if __name__ == "__main__":
    main() 