import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

from ml.preprocessing import ALL_STOPWORDS

def plot_class_distribution(labels: List[int], title: str = "Распределение классов", 
                           save_path: Optional[str] = None):
    """
    Визуализация распределения классов
    
    Args:
        labels: список меток (0/1)
        title: заголовок графика
        save_path: путь для сохранения графика (если None, график будет показан)
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(['Нет депрессии (0)', 'Есть депрессия (1)'], counts, color=['#4CAF50', '#F44336'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    
    plt.title(title)
    plt.ylabel('Количество примеров')
    plt.xlabel('Класс')
    
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]
    
    plt.annotate(f'{percentages[0]:.1f}%', xy=(0, counts[0]/2), ha='center')
    plt.annotate(f'{percentages[1]:.1f}%', xy=(1, counts[1]/2), ha='center')
    
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
    
    # график потерь
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
    
    # график метрик
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

def plot_wordcloud(texts: List[str], 
                  class_labels: Optional[List[int]] = None, 
                  class_to_plot: Optional[int] = None,
                  title: str = "Облако слов",
                  additional_stopwords: Optional[List[str]] = None,
                  max_words: int = 200,
                  save_path: Optional[str] = None,
                  width: int = 800, 
                  height: int = 400):
    """
    Создание облака слов из текстов с учетом стоп-слов
    
    Args:
        texts: список текстов
        class_labels: список меток классов (если нужно строить облако для определенного класса)
        class_to_plot: какой класс визуализировать (0 - не депрессия, 1 - депрессия)
        title: заголовок графика
        additional_stopwords: дополнительные стоп-слова
        max_words: максимальное количество слов в облаке
        save_path: путь для сохранения графика
        width: ширина изображения
        height: высота изображения
    """
    if class_labels is not None and class_to_plot is not None:
        texts = [text for text, label in zip(texts, class_labels) if label == class_to_plot]
        if class_to_plot == 1:
            title = f"{title} (класс: Депрессия)"
        else:
            title = f"{title} (класс: Нет депрессии)"
    
    text = " ".join(texts)
    
    stopwords = set(ALL_STOPWORDS)
    if additional_stopwords:
        stopwords.update(additional_stopwords)
    
    wordcloud = WordCloud(
        width=width, 
        height=height,
        max_words=max_words,
        stopwords=stopwords,
        background_color='white',
        colormap='viridis',
        collocations=False,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.tight_layout(pad=0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_wordcloud_comparison(texts: List[str], 
                             labels: List[int],
                             additional_stopwords: Optional[List[str]] = None,
                             max_words: int = 100,
                             save_dir: Optional[str] = None):
    """
    Создание сравнительных облаков слов для обоих классов
    
    Args:
        texts: список текстов
        labels: список меток классов
        additional_stopwords: дополнительные стоп-слова
        max_words: максимальное количество слов в облаке
        save_dir: директория для сохранения графиков
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    plot_wordcloud(
        texts=texts, 
        class_labels=labels, 
        class_to_plot=0,
        title="Облако слов",
        additional_stopwords=additional_stopwords,
        max_words=max_words,
        save_path=os.path.join(save_dir, 'wordcloud_class0.png') if save_dir else None
    )
    
    plot_wordcloud(
        texts=texts, 
        class_labels=labels, 
        class_to_plot=1,
        title="Облако слов",
        additional_stopwords=additional_stopwords,
        max_words=max_words,
        save_path=os.path.join(save_dir, 'wordcloud_class1.png') if save_dir else None
    )

def plot_top_words(texts: List[str], 
                  labels: List[int], 
                  top_n: int = 20,
                  save_dir: Optional[str] = None):
    """
    Построение графика с наиболее часто встречающимися словами в каждом классе
    
    Args:
        texts: список текстов
        labels: список меток классов
        top_n: количество топ-слов для отображения
        save_dir: директория для сохранения графиков
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    texts_class0 = [text for text, label in zip(texts, labels) if label == 0]
    texts_class1 = [text for text, label in zip(texts, labels) if label == 1]
    
    words_class0 = " ".join(texts_class0).split()
    words_class1 = " ".join(texts_class1).split()
    
    words_class0 = [w for w in words_class0 if w not in ALL_STOPWORDS]
    words_class1 = [w for w in words_class1 if w not in ALL_STOPWORDS]
    
    counter_class0 = Counter(words_class0)
    counter_class1 = Counter(words_class1)
    
    top_words_class0 = [item[0] for item in counter_class0.most_common(top_n)]
    top_freqs_class0 = [item[1] for item in counter_class0.most_common(top_n)]
    
    top_words_class1 = [item[0] for item in counter_class1.most_common(top_n)]
    top_freqs_class1 = [item[1] for item in counter_class1.most_common(top_n)]
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(top_n), top_freqs_class0, color='#4CAF50')
    plt.yticks(range(top_n), top_words_class0)
    plt.xlabel('Частота')
    plt.ylabel('Слова')
    plt.title('Топ-{} слов для класса "Нет депрессии"'.format(top_n))
    plt.gca().invert_yaxis()  # инверсия оси Y для отображения наиболее частых слов сверху
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'top_words_class0.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(top_n), top_freqs_class1, color='#F44336')
    plt.yticks(range(top_n), top_words_class1)
    plt.xlabel('Частота')
    plt.ylabel('Слова')
    plt.title('Топ-{} слов для класса "Есть депрессия"'.format(top_n))
    plt.gca().invert_yaxis()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'top_words_class1.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC кривая')
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
    
    df = pd.DataFrame(meta_features, columns=feature_names)
    df['label'] = labels
    
    for feature in feature_names:
        plt.figure(figsize=(10, 6))
        
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
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if 'history' in results:
        plot_training_history(results['history'], output_dir)
    
    if 'final_metrics' in results:
        metrics = results['final_metrics']
        print(f"Финальные метрики:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
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