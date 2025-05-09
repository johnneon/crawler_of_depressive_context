import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MetaFeaturesEncoder(nn.Module):
    """
    Кодировщик метаданных с глубокой архитектурой и нормализацией
    """
    def __init__(self, meta_dim: int, dropout: float = 0.5):
        """
        Инициализация кодировщика метаданных
        
        Args:
            meta_dim: размерность входных метаданных
            dropout: вероятность дропаута
        """
        super(MetaFeaturesEncoder, self).__init__()
        
        # сеть для обработки метаданных
        self.encoder = nn.Sequential(
            # первый блок
            nn.Linear(meta_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # второй блок
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            
            # третий блок
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            
            # финальный блок
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            x: входные метаданные [batch_size, meta_dim]
            
        Returns:
            torch.Tensor: кодированные метаданные [batch_size, 16]
        """
        return self.encoder(x)

class BiLSTMWithAttention(nn.Module):
    """
    Двунаправленная LSTM с механизмом внимания для классификации текстов
    с учетом метаданных
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, meta_dim: int, 
                 output_dim: int = 1, dropout: float = 0.5, lstm_layers: int = 2):
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
        
        # механизм внимания с масштабированием градиента
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        
        # слои нормализации
        self.text_layernorm = nn.LayerNorm(hidden_dim * 2)
        
        # кодировщик метаданных
        self.meta_encoder = MetaFeaturesEncoder(meta_dim, dropout)

        # классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            
            nn.Linear(64, output_dim)
        )
        
        # применяем инициализацию весов
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
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def attention_net(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Механизм внимания
        
        Args:
            lstm_output: выход LSTM
            
        Returns:
            torch.Tensor: взвешенное представление
        """
        attn_scores = self.attention_weights(lstm_output)
        
        # нормализация для стабильного градиента
        attn_scores = attn_scores / torch.sqrt(torch.tensor(self.hidden_dim * 2, dtype=torch.float32))
        
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_output)
        
        return context.squeeze(1)

    def forward(self, text_embeddings: torch.Tensor, meta_features: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            text_embeddings: эмбеддинги текста
            meta_features: метаданные
            
        Returns:
            torch.Tensor: логиты для классификации
        """
        # обработка текста через LSTM
        batch_size = text_embeddings.size(0)
        h0 = torch.zeros(self.lstm_layers * 2, batch_size, self.hidden_dim).to(text_embeddings.device)
        c0 = torch.zeros(self.lstm_layers * 2, batch_size, self.hidden_dim).to(text_embeddings.device)
        
        lstm_out, _ = self.lstm(text_embeddings, (h0, c0))
        
        # применение механизма внимания
        text_rep = self.attention_net(lstm_out)
        text_rep = self.text_layernorm(text_rep)
        
        # обработка метаданных через улучшенный кодировщик
        meta_rep = self.meta_encoder(meta_features)
        
        # объединение и классификация
        combined = torch.cat((text_rep, meta_rep), dim=1)
        out = self.classifier(combined)
        return out.squeeze(1)

def get_pos_weight(labels: list) -> Tuple[torch.Tensor, torch.device]:
    """
    Рассчитывает веса классов для несбалансированных данных
    
    Args:
        labels: список меток
        
    Returns:
        Tuple: веса классов и устройство (CPU/GPU)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    
    return pos_weight, device 