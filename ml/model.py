import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class BiLSTMWithAttention(nn.Module):
    """
    Двунаправленная LSTM с механизмом внимания для классификации текстов
    с учетом метаданных
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, meta_dim: int, 
                 output_dim: int = 1, dropout: float = 0.5, lstm_layers: int = 1):
        """
        Инициализация модели
        
        Args:
            embedding_dim: размерность эмбеддингов
            hidden_dim: размерность скрытого состояния LSTM
            meta_dim: размерность метаданных
            output_dim: размерность выхода (1 для бинарной классификации)
            dropout: вероятность дропаута
            lstm_layers: количество слоев LSTM
        """
        super(BiLSTMWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # LSTM для обработки текста
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Механизм внимания
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        
        # Слои нормализации
        self.text_layernorm = nn.LayerNorm(hidden_dim * 2)
        self.meta_layernorm = nn.LayerNorm(meta_dim)

        # Расширенные слои для обработки метаданных
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout / 3)
        )

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, output_dim)
        )
        
        # Применяем инициализацию весов
        self._init_weights()

    def _init_weights(self):
        """
        Инициализация весов для улучшения сходимости
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def attention_net(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Механизм внимания
        
        Args:
            lstm_output: выход LSTM
            
        Returns:
            torch.Tensor: взвешенное представление
        """
        attn_scores = self.attention_weights(lstm_output).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = torch.sum(lstm_output * attn_weights, dim=1)
        return context

    def forward(self, text_embeddings: torch.Tensor, meta_features: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            text_embeddings: эмбеддинги текста
            meta_features: метаданные
            
        Returns:
            torch.Tensor: вероятность принадлежности к положительному классу
        """
        # Нормализация метаданных
        meta_features = self.meta_layernorm(meta_features)
        
        # Обработка текста через LSTM
        batch_size = text_embeddings.size(0)
        h0 = torch.zeros(self.lstm_layers * 2, batch_size, self.hidden_dim).to(text_embeddings.device)
        c0 = torch.zeros(self.lstm_layers * 2, batch_size, self.hidden_dim).to(text_embeddings.device)
        lstm_out, _ = self.lstm(text_embeddings, (h0, c0))
        
        # Применение механизма внимания
        text_rep = self.attention_net(lstm_out)
        text_rep = self.text_layernorm(text_rep)
        
        # Обработка метаданных
        meta_rep = self.meta_fc(meta_features)
        
        # Объединение и классификация
        combined = torch.cat((text_rep, meta_rep), dim=1)
        out = self.classifier(combined)
        return torch.sigmoid(out).squeeze(1)

class BiLSTMWithAttentionAndKFold(BiLSTMWithAttention):
    """
    Расширенная версия модели с поддержкой K-fold валидации
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, meta_dim: int, 
                output_dim: int = 1, dropout: float = 0.5, lstm_layers: int = 1):
        super(BiLSTMWithAttentionAndKFold, self).__init__(
            embedding_dim, hidden_dim, meta_dim, output_dim, dropout, lstm_layers
        )
        self.fold_models = []
        
    def add_fold_model(self, model_state: dict):
        """
        Добавить модель для складки
        
        Args:
            model_state: состояние модели
        """
        self.fold_models.append(model_state)
    
    def predict_with_folds(self, text_embeddings: torch.Tensor, meta_features: torch.Tensor) -> torch.Tensor:
        """
        Предсказание с усреднением по всем моделям складок
        
        Args:
            text_embeddings: эмбеддинги текста
            meta_features: метаданные
            
        Returns:
            torch.Tensor: усредненные вероятности
        """
        # Если нет моделей складок, используем обычное предсказание
        if not self.fold_models:
            return self.forward(text_embeddings, meta_features)
        
        # Собираем предсказания от всех моделей
        all_preds = []
        
        # Сохраняем текущее состояние модели
        current_state = self.state_dict()
        
        # Проходим по всем моделям складок
        for fold_state in self.fold_models:
            self.load_state_dict(fold_state)
            with torch.no_grad():
                preds = self.forward(text_embeddings, meta_features)
            all_preds.append(preds.unsqueeze(0))
        
        # Восстанавливаем исходное состояние модели
        self.load_state_dict(current_state)
        
        # Усредняем предсказания
        all_preds = torch.cat(all_preds, dim=0)
        return torch.mean(all_preds, dim=0)

def get_pos_weight(labels: list) -> Tuple[torch.Tensor, torch.device]:
    """
    Рассчитывает веса классов для несбалансированных данных
    
    Args:
        labels: список меток
        
    Returns:
        Tuple: веса классов и устройство (CPU/GPU)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Подсчет количества экземпляров каждого класса
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    
    return pos_weight, device 