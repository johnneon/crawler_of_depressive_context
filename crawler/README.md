# Краулер профилей ВКонтакте

Скрипт для сбора детальной информации о постах и профилях пользователей ВКонтакте с поддержкой двухфакторной аутентификации.

## Функциональные возможности

### Работа с ВКонтакте
- Авторизация на сайте VK с использованием номера телефона/логина и пароля
- Поддержка двухфакторной аутентификации через SMS
- Работа как с русским, так и с английским интерфейсом ВКонтакте

### Сбор данных профилей
- Имя пользователя и статус
- Верификация аккаунта
- Город (извлекается с помощью анализа URL)
- Определение пола пользователя (на основе анализа имени)
- Дата рождения
- Количество подписчиков и друзей

### Анализ постов пользователя
- Извлечение ID поста
- Извлечение текста постов

### Анализ на признаки депрессии
- Автоматическое определение признаков депрессии на основе текстов пользователя
- Использование нейросетевой модели BiLSTM с механизмом внимания
- Учет метаданных пользователя (пол, количество подписчиков и т.д.)
- Вывод вероятности наличия депрессии и итогового прогноза

### Дополнительные особенности
- Обработка закрытых и открытых профилей
- Обработка нескольких пользователей последовательно
- Поддержка как ID пользователей, так и полных URL-адресов профилей
- Сохранение результатов в структурированный JSON-файл
- Гибкая настройка параметров через аргументы командной строки

## Структура проекта

Проект имеет модульную структуру:

```
crawler/
├── main.py                  # Основной модуль скрипта
├── auth/                    # Модуль авторизации
│   ├── __init__.py
│   └── vk_auth.py           # Функции для авторизации в ВКонтакте
├── extractors/              # Модуль извлечения данных
│   ├── __init__.py
│   ├── post_extractor.py    # Извлечение данных о постах
│   └── profile_extractor.py # Извлечение информации о профиле
└── utils/                   # Вспомогательные утилиты
    ├── __init__.py
    ├── browser.py           # Настройка браузера и работа со страницами
    ├── file_ops.py          # Операции с файлами (сохранение данных)
    └── profile_validation.py # Проверка доступности профилей
```

## Установка

Установите зависимости:

```bash
pip install -r requirements.txt
```

## Использование

### Сбор данных пользователей, указанных в командной строке

Можно использовать как ID пользователей, так и полные URL на их профили:

```bash
python -m crawler.main --login ваш_логин --password ваш_пароль --users durov,https://vk.com/id1,anotheruser
```

### Сбор данных пользователей из файла

Создайте текстовый файл со списком ID пользователей или URL-адресов их профилей (по одному на строку):

```
durov
https://vk.com/id123456
anotheruser
```

Затем запустите скрипт с указанием этого файла:

```bash
python -m crawler.main --login ваш_логин --password ваш_пароль --users-file users.txt
```

### Другие параметры

```bash
python -m crawler.main --login ваш_логин --password ваш_пароль --users durov,https://vk.com/id123456 --scrolls 5 --output my_data.json
```

Запуск с видимым GUI (по умолчанию браузер запускается без интерфейса):

```bash
python -m crawler.main --login ваш_логин --password ваш_пароль --users durov --visible
```

### Доступные параметры

- `--login` - Логин или номер телефона пользователя VK (**обязательный параметр**)
- `--password` - Пароль пользователя VK (**обязательный параметр**)
- `--users` - Список ID пользователей или URL профилей через запятую (например: durov,https://vk.com/id1)
- `--users-file` - Путь к файлу со списком ID пользователей или URL профилей (по одному на строку)
- `--scrolls` - Количество прокруток страницы (по умолчанию: 10)
- `--output` - Путь к файлу для сохранения данных (по умолчанию: vk_data.json)
- `--visible` - Запуск браузера с видимым GUI (по умолчанию браузер запускается в headless режиме)
- `--predict-depression` - Активировать предсказание депрессии для каждого пользователя
- `--model-path` - Путь к модели для предсказания депрессии (по умолчанию: models/depression_model.pt)
- `--fasttext-path` - Путь к модели FastText (по умолчанию: cc.ru.300.bin)

**Примечание:** Необходимо указать хотя бы один из параметров `--users` или `--users-file`.

## Формат данных

Скрипт сохраняет данные всех пользователей в один JSON-файл в виде массива:

```json
[
  {
    "name": "Павел Дуров",
    "user_id": "durov",
    "status": "создаю будущее",
    "followers_count": 11112,
    "gender": "male",
    "city": "Saint Petersburg",
    "birthday": "10 октября",
    "posts": [
      {
        "post_id": 2407925,
        "text": "Текст поста..."
      },
      // другие посты...
    ]
  },
  {
    "name": "Другой Пользователь",
    "user_id": "id123456",
    "status": "в сети",
    "followers_count": 500,
    "gender": "female",
    "city": "Moscow",
    "posts": [
      // посты пользователя...
    ]
  },
  // другие пользователи...
]
```

## Особенности работы с двухфакторной аутентификацией

Если на аккаунте включена двухфакторная аутентификация:

1. Скрипт будет запрашивать код из SMS-сообщения в консоли
2. После ввода SMS-кода нужно будет ввести пароль
3. Если появится системное окно "Безопасность Windows" с запросом ключа доступа - нажмите "Отмена"

## Подробное описание модулей

### auth
Модуль отвечает за авторизацию пользователя в ВКонтакте:
- Обработка процесса входа через логин и пароль
- Поддержка двухфакторной аутентификации
- Обработка сессий и перенаправлений

### extractors
Модуль содержит компоненты для извлечения данных:

#### post_extractor.py
Извлекает данные о постах со страницы пользователя:
- `extract_post_id` - извлекает числовой ID поста из атрибута data-post-id
- `extract_post_text` - извлекает текст поста с очисткой от HTML, ссылок и упоминаний

#### profile_extractor.py
Извлекает информацию о профиле пользователя:
- `extract_user_name` - извлекает имя пользователя
- `extract_user_status` - извлекает статус профиля
- `extract_user_city` - извлекает город из профиля и из URL
- `extract_user_gender` - определяет пол пользователя на основе имени
- `extract_user_birthday` - извлекает дату рождения
- `extract_user_followers` - извлекает количество подписчиков и друзей

### utils
Вспомогательные утилиты для работы краулера:
- `browser.py` - настройка экземпляра WebDriver и функции для взаимодействия с браузером
- `file_ops.py` - функции для сохранения данных в файлы JSON
- `profile_validation.py` - проверка доступности профилей (закрытые/несуществующие аккаунты)

## Обработка ошибок

Скрипт обрабатывает следующие сценарии:
- Закрытые профили пользователей
- Несуществующие страницы
- Ошибки при авторизации
- Таймауты при загрузке страниц
- Ошибки при обработке отдельных пользователей (скрипт продолжит работу с другими)
- Корректная работа с разными языками интерфейса (русский/английский)

## Технические особенности

- Требуется Chrome браузер
- Использование Selenium WebDriver для взаимодействия с браузером
- Применение explicit waits для надежного ожидания элементов
- Регулярные выражения для очистки и анализа текста
- Обработка видимого и скрытого режимов браузера
- Модульная архитектура для легкого расширения функциональности 

## Анализ на признаки депрессии

Для использования функциональности анализа на признаки депрессии необходимы:

1. Предварительно обученная модель нейронной сети (BiLSTM с механизмом внимания)
2. Модель FastText для русского языка (cc.ru.300.bin)

### Запуск с предсказанием депрессии

```bash
python -m crawler.main --login ваш_логин --password ваш_пароль --users durov --predict-depression
```

С указанием путей к моделям:

```bash
python -m crawler.main --login ваш_логин --password ваш_пароль --users durov --predict-depression --model-path /путь/к/модели.pt --fasttext-path /путь/к/fasttext.bin
```

### Интерпретация результатов

Результаты анализа включают:

- `probability` - вероятность наличия депрессии (от 0 до 1)
- `has_depression` - булево значение (true/false), указывающее на наличие признаков депрессии
- `label` - метка класса (0 - нет признаков депрессии, 1 - есть признаки депрессии)

**Важно**: Результаты анализа не являются медицинским диагнозом и не могут использоваться для самодиагностики. При подозрении на депрессию следует обратиться к квалифицированному специалисту. 