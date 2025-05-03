# Система анализа депрессивного контекста

Проект содержит инструменты для сбора, анализа и классификации текстовых данных на предмет наличия признаков депрессии.

## Структура проекта

- **ml/** - модуль для обработки и анализа данных с использованием методов машинного обучения
  - **preprocessing.py** - функции для предобработки текстовых данных, обработка выбросов, аугментация текстов
  - **dataset.py** - класс для работы с данными
  - **model.py** - улучшенная архитектура модели с механизмом внимания и многослойной LSTM
  - **balancing.py** - методы балансировки выборок
  - **training.py** - функции для обучения и оценки модели, включая K-fold валидацию
  - **prediction.py** - класс для получения предсказаний
  - **train.py** - скрипт для обучения модели
  - **predict.py** - скрипт для получения предсказаний
  - **visualization.py** - функции для визуализации результатов
  - **visualize_data.py** - скрипт для анализа и визуализации данных
- **dataset/** - директория с данными
- **models/** - директория для сохранения обученных моделей
- **ml_pipeline.py** - обертка для запуска модульного конвейера обработки

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Предварительные требования

### FastText модель

Для работы требуется предобученная модель FastText для русского языка (`cc.ru.300.bin`), которая используется для создания эмбеддингов текста. Эта модель содержит 300-мерные векторные представления слов, обученные на корпусе Common Crawl и Wikipedia.

#### Где скачать модель FastText:

1. Официальный сайт FastText: https://fasttext.cc/docs/en/crawl-vectors.html
2. Прямая ссылка на скачивание: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz

После скачивания распакуйте файл:
```bash
gzip -d cc.ru.300.bin.gz
```

Поместите распакованный файл `cc.ru.300.bin` в корневую директорию проекта или укажите путь к нему при запуске скриптов с помощью параметра `--fasttext_path`.

## Использование

### Визуализация данных

```bash
python -m ml.visualize_data --data_path dataset/data.json --output_dir visualizations/data
```

### Обучение модели

Базовое обучение с балансировкой:
```bash
python -m ml.train --data_path dataset/data.json --fasttext_path cc.ru.300.bin --epochs 20 --balance_method random_oversample
```

Обучение с обработкой выбросов и аугментацией данных:
```bash
python -m ml.train --data_path dataset/data.json --fasttext_path cc.ru.300.bin --epochs 20 --balance_method random_oversample --handle_outliers --augment_positive
```

Обучение с K-fold валидацией:
```bash
python -m ml.train --data_path dataset/data.json --fasttext_path cc.ru.300.bin --epochs 20 --balance_method random_oversample --use_kfold --n_splits 5
```

### Получение предсказаний

```bash
python -m ml.predict --model_path models/depression_model.pt --input_file new_data.json
```

### Визуализация результатов

```bash
python -m ml.visualization --results_file results/results.json --output_dir visualizations/results
```

## Архитектура модели

Система использует двунаправленную LSTM с механизмом внимания для обработки текста и дополнительно учитывает метаданные пользователя:

- Пол
- Количество подписчиков
- Отношение к алкоголю
- Отношение к курению
- Главное в жизни
- Главное в людях

Модель обучается на несбалансированной выборке с использованием различных техник балансировки и оптимизации.

## Метрики

Для оценки качества модели используются следующие метрики:
- Accuracy (точность)
- Precision (точность классификации)
- Recall (полнота)
- F1-мера
- ROC-AUC

## Авторы

Проект разработан для анализа депрессивного контекста в текстовых сообщениях пользователей социальных сетей. 