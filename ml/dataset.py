import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Any

class DepressionDataset(Dataset):
    """
    Класс для представления данных для модели PyTorch
    """
    def __init__(self, texts: List[str], metas: List[List[float]], labels: List[int], 
                 ft_model: Any, max_len: int = 500):
        """
        Инициализация датасета
        
        Args:
            texts: список текстов
            metas: список метаданных
            labels: список меток (0/1)
            ft_model: модель FastText
            max_len: максимальная длина последовательности
        """
        self.texts = texts
        self.metas = metas
        self.labels = labels
        self.ft_model = ft_model
        self.max_len = max_len
        self.embedding_dim = ft_model.get_dimension()
        self.cache = {}

    def __len__(self):
        return len(self.texts)

    def get_embedding(self, tokens: List[str]) -> np.ndarray:
        """
        Получение эмбеддингов токенов с кешированием
        
        Args:
            tokens: список токенов
            
        Returns:
            ndarray: массив эмбеддингов размером (max_len, embedding_dim)
        """
        vecs = []
        for token in tokens:
            if token not in self.cache:
                self.cache[token] = self.ft_model.get_word_vector(token)
            vecs.append(self.cache[token])
        
        # padding/truncation
        if len(vecs) > self.max_len:
            vecs = vecs[:self.max_len]
        else:
            vecs += [np.zeros(self.embedding_dim)] * (self.max_len - len(vecs))
        
        return np.array(vecs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Получение элемента по индексу
        
        Args:
            idx: индекс элемента
            
        Returns:
            dict: словарь с ключами 'text', 'meta', 'label'
        """
        tokens = self.texts[idx].split()
        text_embed = self.get_embedding(tokens)
        meta = np.array(self.metas[idx], dtype=np.float32)
        label = self.labels[idx]
        return {
            'text': torch.tensor(text_embed, dtype=torch.float32),
            'meta': torch.tensor(meta, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

def prepare_data(texts: List[str], meta_features: List[List[float]], labels: List[int], 
                 test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Подготовка данных: нормализация и разделение на обучающую и валидационную выборки
    
    Args:
        texts: список текстов
        meta_features: список метаданных
        labels: список меток
        test_size: доля данных для тестирования
        random_state: случайное состояние для воспроизводимости
        
    Returns:
        tuple: кортеж с данными для обучения и валидации
    """
    # Нормализация мета-признаков
    scaler = MinMaxScaler()
    meta_scaled = scaler.fit_transform(meta_features)
    
    # Разделение на обучающую и валидационную выборки
    return train_test_split(
        texts, meta_scaled, labels, 
        test_size=test_size, 
        stratify=labels,
        random_state=random_state
    ) 