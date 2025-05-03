import argparse
import json
import os
import sys
from typing import List, Dict, Any
import pandas as pd

from ml.prediction import DepressionPredictor

def parse_args():
    """
    Парсинг аргументов командной строки
    """
    parser = argparse.ArgumentParser(description='Получение предсказаний с помощью обученной модели')
    
    parser.add_argument('--model_path', type=str, default='models/depression_model.pt',
                      help='Путь к обученной модели')
    parser.add_argument('--fasttext_path', type=str, default='cc.ru.300.bin',
                      help='Путь к модели FastText')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Путь к файлу с пользователями (JSON)')
    parser.add_argument('--output_file', type=str, default='results/predictions.json',
                      help='Путь для сохранения предсказаний')
    parser.add_argument('--csv_output', type=str, default='results/predictions.csv',
                      help='Путь для сохранения предсказаний в формате CSV')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                      help='Устройство для вычислений (по умолчанию автоматический выбор)')
    
    return parser.parse_args()

def main():
    """
    Основная функция для получения предсказаний
    """
    args = parse_args()
    
    # Создание директории для результатов
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Загрузка данных
    try:
        print(f"Загрузка данных из {args.input_file}...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        if not isinstance(users, list):
            print("Ошибка: данные должны быть в формате списка словарей.")
            sys.exit(1)
            
        print(f"Загружено {len(users)} пользователей.")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)
    
    # Инициализация предсказателя
    try:
        predictor = DepressionPredictor(
            model_path=args.model_path,
            fasttext_path=args.fasttext_path,
            device=args.device
        )
    except Exception as e:
        print(f"Ошибка при инициализации предсказателя: {e}")
        sys.exit(1)
    
    # Получение предсказаний
    print("Получение предсказаний...")
    try:
        predictions = predictor.predict_batch(users)
        print(f"Сделано {len(predictions)} предсказаний.")
        
        # Подсчет статистики
        pos_count = sum(1 for p in predictions if p['prediction'] == 1)
        neg_count = len(predictions) - pos_count
        print(f"Статистика предсказаний: Позитивные (есть депрессия) - {pos_count}, Негативные - {neg_count}")
        
        # Сохранение результатов в JSON
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
        print(f"Результаты сохранены в {args.output_file}")
        
        # Сохранение результатов в CSV
        if args.csv_output:
            df = pd.DataFrame(predictions)
            df.to_csv(args.csv_output, index=False)
            print(f"Результаты сохранены в {args.csv_output}")
        
    except Exception as e:
        print(f"Ошибка при получении предсказаний: {e}")
        sys.exit(1)

def predict_text(text: str, meta_data: Dict[str, int] = None, 
                model_path: str = 'models/depression_model.pt',
                fasttext_path: str = 'cc.ru.300.bin'):
    """
    Функция для предсказания наличия депрессии по тексту
    
    Args:
        text: текст для анализа
        meta_data: словарь с метаданными
        model_path: путь к обученной модели
        fasttext_path: путь к модели FastText
        
    Returns:
        Dict: словарь с предсказанием
    """
    try:
        # Инициализация предсказателя
        predictor = DepressionPredictor(
            model_path=model_path,
            fasttext_path=fasttext_path
        )
        
        # Получение предсказания
        return predictor.predict_from_text(text, meta_data)
    except Exception as e:
        print(f"Ошибка при получении предсказания: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    main() 