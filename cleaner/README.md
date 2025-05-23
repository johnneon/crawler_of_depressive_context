# Модуль очистки, подготовки и разметки данных

## Обзор
Модуль `cleaner` предназначен для обработки и автоматической разметки данных пользователей, полученных из социальных сетей. Основная задача модуля - очистка текстов, идентификация депрессивного контента на основе лексических маркеров и подготовка данных для дальнейшего анализа.

## Основные возможности
- Очистка текста от ссылок, специальных символов и лишних пробелов
- Автоматическая идентификация текстов с признаками депрессии
- Структурирование данных пользователя и его постов
- Анонимизация ID пользователей
- Обработка больших наборов данных из нескольких директорий
- Исключение дубликатов пользователей

## Методы идентификации депрессивного контента

### Категории ключевых слов
Для выявления признаков депрессии в текстовых данных используются пять категорий ключевых слов:

1. **Лексические маркеры**
   - Русские: "бессмысленно", "устал", "надоело", "никому не нужен", "ненавижу себя", "депресс"
   - Английские: "meaningless", "tired", "fed up", "nobody needs me", "hate myself", "depress"

2. **Маркеры изоляции**
   - Русские: "одиночество", "все ушли", "остался один", "никого рядом", "изолирован" 
   - Английские: "loneliness", "everyone left", "left alone", "nobody around", "isolated"

3. **Самоориентированные маркеры**
   - Русские: "я пустой", "я не могу", "мне плохо", "не вижу смысла"
   - Английские: "i'm empty", "i can't", "i feel bad", "see no point", "no purpose"

4. **Структурные маркеры**
   - Русские: "жить. больно.", "нет сил", "нет смысла"
   - Английские: "living hurts", "no strength", "no meaning", "pointless"

5. **Физиологические маркеры**
   - Русские: "не могу спать", "нет аппетита", "встать с кровати", "не хочу есть"
   - Английские: "can't sleep", "no appetite", "get out of bed", "don't want to eat"

Система помечает пользователя как "депрессивного" (метка 1), если в его постах обнаружено хотя бы одно ключевое слово из любой категории. В противном случае пользователю присваивается метка 0.

## Алгоритм обработки данных

1. **Очистка текста постов**
2. **Определение депрессивного контекста**
3. **Очистка и структурирование данных пользователя**
4. **Анонимизация ID пользователей**

## Использование

### Базовое использование
```bash
python main.py
```
По умолчанию скрипт будет искать и обрабатывать файлы JSON в директории `data/`.

### Указание нескольких директорий
```bash
python main.py dir1 dir2 dir3
```
Скрипт последовательно обработает все JSON-файлы в указанных директориях, исключая дубликаты пользователей.

### Результаты
Результаты обработки сохраняются в файле `dataset/data.json`, содержащем:
- Очищенные и структурированные данные пользователей
- Очищенные тексты постов
- Метки депрессивного контента (1 - депрессия, 0 - норма)
- Анонимизированные ID пользователей

## Статистика
В процессе работы скрипт выводит информацию о:
- Количестве обработанных директорий и файлов
- Общем количестве обработанных уникальных пользователей
- Количестве и проценте пользователей с депрессивным контентом

## Требования
- Python 3.7+
- pandas
- tqdm 