import torch
import numpy as np
import fasttext
from typing import Dict, Any, List

from ml.model import BiLSTMWithAttention
from ml.preprocessing import preprocess_user, clean_text, lemmatize

class DepressionPredictor:
    """
    Класс для предсказания депрессии у пользователей
    """
    def __init__(self, model_path: str, fasttext_path: str = "cc.ru.300.bin", 
                 device: str = None, max_len: int = 500):
        """
        Инициализация предсказателя
        
        Args:
            model_path: путь к сохраненной модели
            fasttext_path: путь к модели FastText
            device: устройство (gpu/cpu)
            max_len: максимальная длина последовательности
        """
        # Определение устройства
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Загрузка модели FastText
        print(f"Загрузка модели FastText из {fasttext_path}...")
        self.ft_model = fasttext.load_model(fasttext_path)
        
        # Параметры модели
        self.embedding_dim = 300  # размерность FastText
        self.hidden_dim = 128
        self.meta_dim = 6  # количество мета признаков
        self.max_len = max_len
        
        # Инициализация модели
        print(f"Инициализация модели из {model_path}...")
        self.model = BiLSTMWithAttention(
            self.embedding_dim, self.hidden_dim, self.meta_dim
        ).to(self.device)
        
        # Загрузка весов модели
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Кеш для эмбеддингов
        self.embedding_cache = {}
        
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Получение эмбеддингов текста
        
        Args:
            text: текст для получения эмбеддингов
            
        Returns:
            np.ndarray: массив эмбеддингов
        """
        tokens = text.split()
        vecs = []
        
        for token in tokens:
            if token not in self.embedding_cache:
                self.embedding_cache[token] = self.ft_model.get_word_vector(token)
            vecs.append(self.embedding_cache[token])
        
        # padding/truncation
        if len(vecs) > self.max_len:
            vecs = vecs[:self.max_len]
        else:
            vecs += [np.zeros(self.embedding_dim)] * (self.max_len - len(vecs))
        
        return np.array(vecs)
    
    def predict_user(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Предсказание для пользователя
        
        Args:
            user: словарь с данными пользователя
            
        Returns:
            Dict: словарь с предсказанием и вероятностью
        """
        # Предобработка пользователя
        processed_user = preprocess_user(user)
        
        # Получение эмбеддингов текста
        text_embed = self.get_embedding(processed_user['text'])
        text_tensor = torch.tensor(text_embed, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Получение метаданных
        meta_data = list(processed_user['meta'].values())
        meta_tensor = torch.tensor(meta_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            prob = self.model(text_tensor, meta_tensor).item()
        
        # Определение класса
        pred_class = 1 if prob >= 0.5 else 0
        
        return {
            'user_id': user.get('user_id', ''),
            'probability': prob,
            'prediction': pred_class,
            'has_depression': bool(pred_class)
        }
    
    def predict_batch(self, users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Предсказание для списка пользователей
        
        Args:
            users: список словарей с данными пользователей
            
        Returns:
            List[Dict]: список словарей с предсказаниями
        """
        return [self.predict_user(user) for user in users]
    
    def predict_from_text(self, text: str, meta_data: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Предсказание по тексту и метаданным
        
        Args:
            text: текст
            meta_data: словарь с метаданными
            
        Returns:
            Dict: словарь с предсказанием и вероятностью
        """
        # Если метаданные не предоставлены, используем нули
        if meta_data is None:
            meta_data = {
                "sex": 0,
                "followers_count": 0,
                "alcohol": 0,
                "smoking": 0,
                "life_main": 0,
                "people_main": 0
            }
        
        # Очистка и лемматизация текста
        cleaned_text = clean_text(text)
        lemmatized_text = lemmatize(cleaned_text)
        
        # Получение эмбеддингов текста
        text_embed = self.get_embedding(lemmatized_text)
        text_tensor = torch.tensor(text_embed, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Получение метаданных
        meta_values = list(meta_data.values())
        meta_tensor = torch.tensor(meta_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            prob = self.model(text_tensor, meta_tensor).item()
        
        # Определение класса
        pred_class = 1 if prob >= 0.5 else 0
        
        return {
            'probability': prob,
            'prediction': pred_class,
            'has_depression': bool(pred_class)
        } 