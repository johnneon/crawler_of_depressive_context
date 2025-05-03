import re
from pymorphy3 import MorphAnalyzer
import numpy as np
from typing import Dict, List, Any, Union, Tuple

morph = MorphAnalyzer()

def clean_text(text: str) -> str:
    """
    Очистка текста от HTML-тегов, ссылок и лишних пробелов
    """
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize(text: str) -> str:
    """
    Лемматизация текста с помощью pymorphy3
    """
    return ' '.join([morph.parse(word)[0].normal_form for word in text.split()])

def handle_outliers(meta_features: List[Dict[str, Union[int, float]]]) -> List[Dict[str, Union[int, float]]]:
    """
    Обработка выбросов в метаданных
    
    Args:
        meta_features: список словарей с метаданными пользователей
        
    Returns:
        List[Dict]: список обработанных метаданных
    """
    # Создаем копию метаданных
    processed_meta = []
    
    # Получаем имена числовых полей
    numeric_fields = []
    for field in meta_features[0].keys():
        if isinstance(meta_features[0][field], (int, float)):
            numeric_fields.append(field)
    
    # Вычисляем статистики для каждого поля
    stats = {}
    for field in numeric_fields:
        values = [user[field] for user in meta_features]
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        median = np.median(values)
        
        stats[field] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'median': median
        }
    
    # Обработка каждого пользователя
    for user in meta_features:
        processed_user = user.copy()
        
        # Обработка каждого поля
        for field in numeric_fields:
            # Если значение является выбросом, заменяем его медианой
            if (user[field] < stats[field]['lower_bound'] or 
                user[field] > stats[field]['upper_bound']):
                processed_user[field] = stats[field]['median']
        
        processed_meta.append(processed_user)
    
    return processed_meta

def preprocess_user(user: dict) -> dict:
    """
    Предобработка данных пользователя: объединение текстов постов, 
    очистка и лемматизация текста
    """
    combined_text = ' '.join(post['text'] for post in user['posts'] if post['text'])
    cleaned_text = clean_text(combined_text)
    lemmatized_text = lemmatize(cleaned_text)
    
    return {
        "label": user["label"],
        "text": lemmatized_text,
        "meta": {
            "sex": user.get("sex", 0),
            "followers_count": user.get("followers_count", 0),
            "alcohol": user.get("alcohol", 0) or 0,
            "smoking": user.get("smoking", 0) or 0,
            "life_main": user.get("life_main", 0) or 0,
            "people_main": user.get("people_main", 0) or 0,
        }
    }

def augment_text(text: str, augmentation_ratio: float = 0.1) -> List[str]:
    """
    Аугментация текста для увеличения количества обучающих примеров
    
    Args:
        text: исходный текст
        augmentation_ratio: доля слов для удаления/перемешивания
        
    Returns:
        List[str]: список аугментированных текстов
    """
    words = text.split()
    augmented_texts = []
    
    if len(words) <= 3:
        return [text]  # Не аугментируем слишком короткие тексты
    
    # 1. Удаление случайных слов
    num_to_remove = max(1, int(len(words) * augmentation_ratio))
    for _ in range(2):
        indices_to_remove = np.random.choice(len(words), num_to_remove, replace=False)
        new_words = [w for i, w in enumerate(words) if i not in indices_to_remove]
        augmented_texts.append(" ".join(new_words))
    
    # 2. Перемешивание порядка некоторых слов
    num_to_shuffle = max(2, int(len(words) * augmentation_ratio))
    for _ in range(2):
        indices_to_shuffle = np.random.choice(len(words), num_to_shuffle, replace=False)
        shuffle_words = [words[i] for i in indices_to_shuffle]
        np.random.shuffle(shuffle_words)
        
        new_words = words.copy()
        for i, idx in enumerate(indices_to_shuffle):
            new_words[idx] = shuffle_words[i]
        
        augmented_texts.append(" ".join(new_words))
    
    return augmented_texts

def preprocess_batch(users: List[Dict[str, Any]], 
                    augment_positive: bool = True,
                    handle_meta_outliers: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Предобработка партии пользователей с дополнительными улучшениями
    
    Args:
        users: список словарей с данными пользователей
        augment_positive: нужно ли аугментировать положительные примеры
        handle_meta_outliers: нужно ли обрабатывать выбросы в метаданных
        
    Returns:
        Tuple: кортеж (обработанные пользователи, статистика)
    """
    processed_users = []
    stats = {
        'original_count': len(users),
        'positive_count': 0,
        'negative_count': 0,
        'augmented_count': 0
    }
    
    # Предобработка пользователей
    basic_processed = [preprocess_user(user) for user in users]
    
    # Обработка выбросов в метаданных
    if handle_meta_outliers:
        meta_list = [user['meta'] for user in basic_processed]
        processed_meta = handle_outliers(meta_list)
        
        for i, user in enumerate(basic_processed):
            user['meta'] = processed_meta[i]
    
    # Аугментация положительных примеров
    for user in basic_processed:
        if user['label'] == 1:
            stats['positive_count'] += 1
            processed_users.append(user)
            
            # Аугментация текстов положительных примеров
            if augment_positive:
                augmented_texts = augment_text(user['text'])
                for aug_text in augmented_texts:
                    augmented_user = user.copy()
                    augmented_user['text'] = aug_text
                    processed_users.append(augmented_user)
                    stats['augmented_count'] += 1
        else:
            stats['negative_count'] += 1
            processed_users.append(user)
    
    return processed_users, stats 