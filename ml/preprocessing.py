import re
from pymorphy3 import MorphAnalyzer

morph = MorphAnalyzer()

def clean_text(text: str) -> str:
    """
    Очистка текста от HTML-тегов, ссылок и лишних пробелов
    """
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize(text: str) -> str:
    """
    Лемматизация текста с помощью pymorphy3
    """
    return ' '.join([morph.parse(word)[0].normal_form for word in text.split()])

def preprocess_user(user: dict) -> dict:
    """
    Предобработка данных пользователя: объединение текстов постов, 
    очистка и лемматизация текста
    """
    combined_text = ' '.join(post['text'] for post in user['posts'] if post['text'])
    cleaned_text = clean_text(combined_text)
    lemmatized_text = lemmatize(cleaned_text)
    
    return {
        "label": user["label"],
        "text": lemmatized_text,
        "meta": {
            "sex": user.get("sex", 0),
            "followers_count": user.get("followers_count", 0),
            "alcohol": user.get("alcohol", 0) or 0,
            "smoking": user.get("smoking", 0) or 0,
            "life_main": user.get("life_main", 0) or 0,
            "people_main": user.get("people_main", 0) or 0,
        }
    } 