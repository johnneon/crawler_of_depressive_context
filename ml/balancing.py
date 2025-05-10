import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from typing import List, Tuple, Dict

def balance_dataset(texts: List[str], meta_features: np.ndarray, 
                    labels: List[int], 
                    method: str = 'none',
                    imbalance_ratio: float = 0.333,  # 1:3 соотношение
                    random_state: int = 42) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Балансировка датасета с использованием различных методов
    
    Args:
        texts: список текстов
        meta_features: массив метаданных
        labels: список меток
        method: метод балансировки ('random_oversample', 'random_undersample', 'smote', 'adasyn', 'partial_smote', 'partial_adasyn' или 'none')
        imbalance_ratio: для частичной балансировки, целевое соотношение между позитивным и негативным классами (например, 0.333 соответствует соотношению 1:3)
        random_state: случайное состояние для воспроизводимости
        
    Returns:
        Tuple: кортеж (тексты, метаданные, метки) после балансировки
    """
    text_indices = np.arange(len(texts)).reshape(-1, 1)
    if method == 'none':
        return texts, meta_features, labels
    
    num_positive = sum(labels)
    num_negative = len(labels) - num_positive
    
    if method.startswith('partial_'):
        base_method = method.replace('partial_', '')
        
        target_positive = int(num_negative * imbalance_ratio)
        
        # если уже больше, не нужно балансировать
        if num_positive >= target_positive:
            return texts, meta_features, labels
        
        sampling_strategy = {1: target_positive}
        
        if base_method == 'smote':
            try:
                sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
                X_res, y_res = sampler.fit_resample(meta_features, labels)
                
                X_res_indices, y_res_indices = sampler.fit_resample(text_indices, labels)
                texts_res = [texts[i[0]] if i[0] < len(texts) else "synthetic_example" for i in X_res_indices]
                
                return texts_res, X_res, y_res.tolist()
            except Exception as e:
                print(f"Ошибка при выполнении частичного SMOTE: {e}")
                return texts, meta_features, labels
                
        elif base_method == 'adasyn':
            try:
                sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
                X_res, y_res = sampler.fit_resample(meta_features, labels)
                
                X_res_indices, y_res_indices = sampler.fit_resample(text_indices, labels)
                texts_res = [texts[i[0]] if i[0] < len(texts) else "synthetic_example" for i in X_res_indices]
                
                return texts_res, X_res, y_res.tolist()
            except Exception as e:
                print(f"Ошибка при выполнении частичного ADASYN: {e}")
                return texts, meta_features, labels
                
    if method == 'random_oversample':
        sampler = RandomOverSampler(random_state=random_state)
        X_res_indices, y_res = sampler.fit_resample(text_indices, labels)
    elif method == 'random_undersample':
        sampler = RandomUnderSampler(random_state=random_state)
        X_res_indices, y_res = sampler.fit_resample(text_indices, labels)
    elif method == 'smote':
        # для SMOTE нужны численные признаки, поэтому используем meta_features
        if meta_features.shape[1] == 0:
            raise ValueError("SMOTE требует численных признаков (meta_features)")
        
        try:
            sampler = SMOTE(random_state=random_state)
            X_res, y_res = sampler.fit_resample(meta_features, labels)
            
            X_res_indices, y_res_indices = sampler.fit_resample(text_indices, labels)
            texts_res = [texts[i[0]] if i[0] < len(texts) else "synthetic_example" for i in X_res_indices]
            
            return texts_res, X_res, y_res.tolist()
        except Exception as e:
            print(f"Ошибка при выполнении SMOTE: {e}, используем RandomOverSampler")
            sampler = RandomOverSampler(random_state=random_state)
            X_res_indices, y_res = sampler.fit_resample(text_indices, labels)
    elif method == 'adasyn':
        # для ADASYN также нужны численные признаки
        if meta_features.shape[1] == 0:
            raise ValueError("ADASYN требует численных признаков (meta_features)")
        
        try:
            sampler = ADASYN(random_state=random_state)
            X_res, y_res = sampler.fit_resample(meta_features, labels)
            
            X_res_indices, y_res_indices = sampler.fit_resample(text_indices, labels)
            texts_res = [texts[i[0]] if i[0] < len(texts) else "synthetic_example" for i in X_res_indices]
            
            return texts_res, X_res, y_res.tolist()
        except Exception as e:
            print(f"Ошибка при выполнении ADASYN: {e}, используем RandomOverSampler")
            sampler = RandomOverSampler(random_state=random_state)
            X_res_indices, y_res = sampler.fit_resample(text_indices, labels)
    else:
        raise ValueError(f"Неизвестный метод балансировки: {method}")
    
    # восстанавливаем данные по индексам
    texts_res = [texts[i[0]] for i in X_res_indices]
    
    meta_features_res = np.array([meta_features[i[0]] for i in X_res_indices])
    
    if isinstance(y_res, np.ndarray):
        y_res = y_res.tolist()
    
    return texts_res, meta_features_res, y_res

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