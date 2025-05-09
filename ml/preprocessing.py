import re
from pymorphy3 import MorphAnalyzer
import numpy as np
from typing import Dict, List, Any, Union, Tuple
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

morph = MorphAnalyzer()

# Получаем стандартные стоп-слова для русского языка
STOPWORDS = set(stopwords.words('russian'))

# Добавляем специфические стоп-слова для соцсетей
SOCIAL_STOPWORDS = {
    'привет', 'пока', 'доброе', 'утро', 'день', 'вечер', 'ночь',
    'http', 'https', 'www', 'com', 'ru', 'фото', 'видео',
    'лайк', 'репост', 'ретвит', 'подписка', 'подписаться', 'отписаться',
    'rt', 'vk', 'ok', 'instagram', 'facebook', 'twitter', 'telegram',
    'спасибо', 'пожалуйста', 'всем', 'хорошего', 'отличного'
}

# Объединяем стандартные и специфические стоп-слова
ALL_STOPWORDS = STOPWORDS.union(SOCIAL_STOPWORDS)

def clean_text(text: str) -> str:
    """
    Очистка текста от HTML-тегов, ссылок и лишних пробелов
    """
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)  # удаление упоминаний (@username)
    text = re.sub(r'#\w+', '', text)  # удаление хештегов (#topic)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text: str) -> str:
    """
    Удаление стоп-слов из текста
    """
    words = text.split()
    filtered_words = [word for word in words if word not in ALL_STOPWORDS]
    return ' '.join(filtered_words)

def lemmatize(text: str) -> str:
    """
    Лемматизация текста с помощью pymorphy3
    """
    return ' '.join([morph.parse(word)[0].normal_form for word in text.split()])

def handle_outliers(meta_features: List[Dict[str, Union[int, float]]]) -> List[Dict[str, Union[int, float]]]:
    """
    Обработка выбросов в метаданных - заменяет выбросы на медианные значения
    
    Args:
        meta_features: список словарей с метаданными пользователей
        
    Returns:
        List[Dict]: список обработанных метаданных
    """
    processed_meta = []
    
    numeric_fields = []
    for field in meta_features[0].keys():
        if isinstance(meta_features[0][field], (int, float)):
            numeric_fields.append(field)
    
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
    
    for user in meta_features:
        processed_user = user.copy()
        
        for field in numeric_fields:
            if (user[field] < stats[field]['lower_bound'] or 
                user[field] > stats[field]['upper_bound']):
                processed_user[field] = stats[field]['median']
        
        processed_meta.append(processed_user)
    
    return processed_meta

def preprocess_user(user: dict) -> dict:
    """
    Предобработка данных пользователя: объединение текстов постов, 
    очистка, удаление стоп-слов и лемматизация текста
    """
    combined_text = ' '.join(post['text'] for post in user['posts'] if post['text'])
    cleaned_text = clean_text(combined_text)
    filtered_text = remove_stopwords(cleaned_text)
    lemmatized_text = lemmatize(filtered_text)
    
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
    
    # не аугментируем слишком короткие тексты
    if len(words) <= 3:
        return [text]
    
    num_to_remove = max(1, int(len(words) * augmentation_ratio))
    for _ in range(2):
        indices_to_remove = np.random.choice(len(words), num_to_remove, replace=False)
        new_words = [w for i, w in enumerate(words) if i not in indices_to_remove]
        augmented_texts.append(" ".join(new_words))
    
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
    Предобработка партии пользователей
    
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
    
    basic_processed = [preprocess_user(user) for user in users]
    
    if handle_meta_outliers:
        meta_list = [user['meta'] for user in basic_processed]
        processed_meta = handle_outliers(meta_list)
        
        for i, user in enumerate(basic_processed):
            user['meta'] = processed_meta[i]
    
    for user in basic_processed:
        if user['label'] == 1:
            stats['positive_count'] += 1
            processed_users.append(user)
            
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