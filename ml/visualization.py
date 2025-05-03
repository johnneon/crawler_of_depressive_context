import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns

def plot_class_distribution(labels: List[int], title: str = "Распределение классов", 
                           save_path: Optional[str] = None):
    """
    Визуализация распределения классов
    
    Args:
        labels: список меток (0/1)
        title: заголовок графика
        save_path: путь для сохранения графика (если None, график будет показан)
    """
    # Подсчет классов
    unique, counts = np.unique(labels, return_counts=True)
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(['Нет депрессии (0)', 'Есть депрессия (1)'], counts, color=['#4CAF50', '#F44336'])
    
    # Добавление значений над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    
    plt.title(title)
    plt.ylabel('Количество примеров')
    plt.xlabel('Класс')
    
    # Добавление процентного соотношения
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]
    
    plt.annotate(f'{percentages[0]:.1f}%', xy=(0, counts[0]/2), ha='center')
    plt.annotate(f'{percentages[1]:.1f}%', xy=(1, counts[1]/2), ha='center')
    
    # Сохранение или отображение
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_history(history: List[Dict[str, float]], output_dir: str = 'results'):
    """
    Визуализация истории обучения
    
    Args:
        history: история обучения
        output_dir: директория для сохранения графиков
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Построение графика потерь
    plt.figure(figsize=(10, 6))
    plt.plot([x['epoch'] for x in history], [x['train_loss'] for x in history], label='Train Loss')
    plt.plot([x['epoch'] for x in history], [x['loss'] for x in history], label='Val Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('Потери при обучении и валидации')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'), bbox_inches='tight')
    plt.close()
    
    # Построение графика метрик
    plt.figure(figsize=(10, 6))
    plt.plot([x['epoch'] for x in history], [x['accuracy'] for x in history], label='Accuracy')
    plt.plot([x['epoch'] for x in history], [x['precision'] for x in history], label='Precision')
    plt.plot([x['epoch'] for x in history], [x['recall'] for x in history], label='Recall')
    plt.plot([x['epoch'] for x in history], [x['f1'] for x in history], label='F1')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    plt.title('Метрики на валидационной выборке')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'), bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true: List[int], y_prob: List[float], save_path: Optional[str] = None):
    """
    Построение ROC-кривой
    
    Args:
        y_true: истинные метки
        y_prob: вероятности положительного класса
        save_path: путь для сохранения графика
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curve(y_true: List[int], y_prob: List[float], save_path: Optional[str] = None):
    """
    Построение кривой точности-полноты
    
    Args:
        y_true: истинные метки
        y_prob: вероятности положительного класса
        save_path: путь для сохранения графика
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Добавление линии случайного классификатора
    pos_rate = sum(y_true) / len(y_true)
    plt.axhline(y=pos_rate, color='gray', linestyle='--')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: Optional[str] = None):
    """
    Построение матрицы ошибок
    
    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        save_path: путь для сохранения графика
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Нет депрессии', 'Есть депрессия'],
               yticklabels=['Нет депрессии', 'Есть депрессия'])
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.title('Матрица ошибок')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_meta_feature_distributions(meta_features: List[List[float]], labels: List[int], 
                                   feature_names: List[str], save_dir: Optional[str] = None):
    """
    Визуализация распределения мета-признаков в зависимости от класса
    
    Args:
        meta_features: список мета-признаков
        labels: список меток
        feature_names: имена признаков
        save_dir: директория для сохранения графиков
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Преобразуем данные в DataFrame
    df = pd.DataFrame(meta_features, columns=feature_names)
    df['label'] = labels
    
    # Для каждого признака строим распределение
    for feature in feature_names:
        plt.figure(figsize=(10, 6))
        
        # Гистограммы для каждого класса
        sns.histplot(data=df, x=feature, hue='label', multiple='dodge', 
                   palette={0: '#4CAF50', 1: '#F44336'})
        
        plt.title(f'Распределение признака "{feature}" по классам')
        plt.xlabel(feature)
        plt.ylabel('Количество')
        plt.legend(['Нет депрессии', 'Есть депрессия'])
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{feature}_distribution.png'), bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def visualize_results_from_file(results_file: str, output_dir: str = 'visualizations'):
    """
    Визуализация результатов из файла
    
    Args:
        results_file: путь к файлу с результатами
        output_dir: директория для сохранения графиков
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Загрузка результатов
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Построение графиков истории обучения
    if 'history' in results:
        plot_training_history(results['history'], output_dir)
    
    # Печать метрик
    if 'final_metrics' in results:
        metrics = results['final_metrics']
        print(f"Финальные метрики:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Сохранение метрик в текстовый файл
        with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
            f.write("Финальные метрики:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Визуализация результатов модели')
    parser.add_argument('--results_file', type=str, default='results/results.json',
                      help='Путь к файлу с результатами')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                      help='Директория для сохранения графиков')
    
    args = parser.parse_args()
    
    visualize_results_from_file(args.results_file, args.output_dir) 