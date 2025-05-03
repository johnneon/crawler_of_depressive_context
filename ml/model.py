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
                 output_dim: int = 1, dropout: float = 0.5):
        """
        Инициализация модели
        
        Args:
            embedding_dim: размерность эмбеддингов
            hidden_dim: размерность скрытого состояния LSTM
            meta_dim: размерность метаданных
            output_dim: размерность выхода (1 для бинарной классификации)
            dropout: вероятность дропаута
        """
        super(BiLSTMWithAttention, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM для обработки текста
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)

        # Слои для обработки метаданных
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, 32),  # Увеличил размерность с 16 до 32
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, 128),  # Увеличил размерность с 64 до 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, output_dim)
        )

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
        lstm_out, _ = self.lstm(text_embeddings)
        text_rep = self.attention_net(lstm_out)
        meta_rep = self.meta_fc(meta_features)
        combined = torch.cat((text_rep, meta_rep), dim=1)
        out = self.classifier(combined)
        return torch.sigmoid(out).squeeze(1)

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