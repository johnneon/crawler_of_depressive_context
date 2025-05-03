import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from typing import List, Tuple, Dict, Any, Optional

def balance_dataset(texts: List[str], meta_features: np.ndarray, 
                    labels: List[int], 
                    method: str = 'random_oversample',
                    random_state: int = 42) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Балансировка датасета с использованием различных методов
    
    Args:
        texts: список текстов
        meta_features: массив метаданных
        labels: список меток
        method: метод балансировки ('random_oversample', 'random_undersample', 'smote' или 'none')
        random_state: случайное состояние для воспроизводимости
        
    Returns:
        Tuple: кортеж (тексты, метаданные, метки) после балансировки
    """
    # Преобразуем тексты в индексы
    text_indices = np.arange(len(texts)).reshape(-1, 1)
    
    if method == 'none':
        return texts, meta_features, labels
    
    # Выбор метода балансировки
    if method == 'random_oversample':
        sampler = RandomOverSampler(random_state=random_state)
        X_res_indices, y_res = sampler.fit_resample(text_indices, labels)
    elif method == 'random_undersample':
        sampler = RandomUnderSampler(random_state=random_state)
        X_res_indices, y_res = sampler.fit_resample(text_indices, labels)
    elif method == 'smote':
        # Для SMOTE нужны численные признаки, поэтому используем meta_features
        if meta_features.shape[1] == 0:
            raise ValueError("SMOTE требует численных признаков (meta_features)")
        sampler = SMOTE(random_state=random_state)
        try:
            X_res, y_res = sampler.fit_resample(meta_features, labels)
            # Восстанавливаем тексты по индексам
            texts_res = [texts[i[0]] for i in X_res_indices]
            return texts_res, X_res, y_res.tolist()
        except:
            print("Ошибка при выполнении SMOTE, используем RandomOverSampler")
            sampler = RandomOverSampler(random_state=random_state)
            X_res_indices, y_res = sampler.fit_resample(text_indices, labels)
    else:
        raise ValueError(f"Неизвестный метод балансировки: {method}")
    
    # Восстанавливаем тексты по индексам
    texts_res = [texts[i[0]] for i in X_res_indices]
    
    # Восстанавливаем meta_features по индексам
    meta_features_res = np.array([meta_features[i[0]] for i in X_res_indices])
    
    return texts_res, meta_features_res, y_res.tolist()

def get_class_weights(labels: List[int]) -> Dict[int, float]:
    """
    Расчет весов классов для несбалансированных данных
    
    Args:
        labels: список меток
        
    Returns:
        Dict: словарь с весами классов
    """
    classes = np.unique(labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    class_weights = {}
    for c in classes:
        class_weights[c] = total_samples / (len(classes) * class_counts[c])
    
    return class_weights 