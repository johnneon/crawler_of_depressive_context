# Модуль для анализа депрессивного контекста

Данный модуль содержит компоненты для обучения и использования модели глубокого обучения для определения признаков депрессии на основе текстовых данных пользователей.

## Структура модуля

- **preprocessing.py** - содержит функции для предобработки текстовых данных (очистка, лемматизация, обработка выбросов, аугментация текста)
- **dataset.py** - определяет класс датасета и функции для подготовки данных
- **model.py** - архитектура модели (BiLSTM с механизмом внимания, слоями нормализации и возможностью использовать многослойную LSTM)
- **balancing.py** - функции для балансировки данных (особенно для несбалансированных классов)
- **training.py** - функции для обучения и оценки модели, включая K-fold валидацию
- **prediction.py** - класс для получения предсказаний с использованием обученной модели
- **train.py** - скрипт для запуска обучения модели
- **predict.py** - скрипт для получения предсказаний
- **visualization.py** - визуализация результатов обучения и предсказаний
- **visualize_data.py** - анализ исходных данных и визуализация распределений

## Предварительные требования

### FastText модель

Для работы с текстом необходима предобученная модель FastText для русского языка `cc.ru.300.bin`. Эта модель содержит векторные представления слов обученные на Common Crawl и Wikipedia для русского языка.

#### Где скачать:

1. Официальный сайт FastText: https://fasttext.cc/docs/en/crawl-vectors.html
2. Прямая ссылка: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz

После скачивания необходимо распаковать файл:
```bash
gzip -d cc.ru.300.bin.gz
```

Поместите файл `cc.ru.300.bin` в корневую директорию проекта или укажите путь к нему при запуске скриптов через параметр `--fasttext_path`.

## Обучение модели

Для обучения модели выполните следующую команду:

```bash
python -m ml.train --data_path dataset/data.json --fasttext_path cc.ru.300.bin --model_dir models --model_name depression_model.pt --epochs 20 --batch_size 32 --balance_method random_oversample
```

Для использования всех улучшений:

```bash
python -m ml.train --data_path dataset/data.json --fasttext_path cc.ru.300.bin --epochs 20 --handle_outliers --augment_positive --use_kfold --n_splits 5 --lstm_layers 2 --dropout 0.5
```

### Параметры для обучения

- `--data_path` - путь к файлу с данными
- `--fasttext_path` - путь к модели FastText
- `--model_dir` - директория для сохранения моделей
- `--model_name` - имя файла модели
- `--epochs` - количество эпох обучения
- `--batch_size` - размер батча
- `--hidden_dim` - размерность скрытого состояния LSTM
- `--dropout` - вероятность дропаута
- `--learning_rate` - скорость обучения
- `--patience` - количество эпох без улучшения до остановки
- `--test_size` - доля данных для валидации
- `--balance_method` - метод балансировки данных (`none`, `random_oversample`, `random_undersample`, `smote`)
- `--max_len` - максимальная длина последовательности
- `--output_dir` - директория для результатов
- `--lstm_layers` - количество слоёв LSTM
- `--use_kfold` - использовать K-fold валидацию
- `--n_splits` - количество разбиений для K-fold валидации
- `--handle_outliers` - обрабатывать выбросы в метаданных
- `--augment_positive` - аугментировать положительные примеры

## Получение предсказаний

Для получения предсказаний с использованием обученной модели:

```bash
python -m ml.predict --model_path models/depression_model.pt --fasttext_path cc.ru.300.bin --input_file new_users.json --output_file results/predictions.json
```

### Параметры для предсказания

- `--model_path` - путь к обученной модели
- `--fasttext_path` - путь к модели FastText
- `--input_file` - путь к файлу с пользователями (JSON)
- `--output_file` - путь для сохранения предсказаний
- `--csv_output` - путь для сохранения предсказаний в формате CSV
- `--device` - устройство для вычислений (`cpu`, `cuda`)

## Использование в коде

Для использования модуля в своем коде можно использовать класс `DepressionPredictor`:

```python
from ml.prediction import DepressionPredictor

# Инициализация предсказателя
predictor = DepressionPredictor(
    model_path='models/depression_model.pt',
    fasttext_path='cc.ru.300.bin'
)

# Предсказание по тексту
text = "Мне кажется, что жизнь потеряла смысл. Ничего не приносит радости."
meta_data = {
    "sex": 1,                  # пол
    "followers_count": 150,    # количество подписчиков
    "alcohol": 2,              # отношение к алкоголю
    "smoking": 1,              # отношение к курению
    "life_main": 3,            # главное в жизни
    "people_main": 2           # главное в людях
}

result = predictor.predict_from_text(text, meta_data)
print(f"Вероятность депрессии: {result['probability']:.2f}")
print(f"У пользователя депрессия: {'Да' if result['has_depression'] else 'Нет'}")
```

## Требования

- Python 3.7+
- PyTorch
- fasttext
- scikit-learn
- imbalanced-learn
- pymorphy3
- numpy
- pandas
- matplotlib 