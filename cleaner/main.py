import re
import json
import pandas as pd
import os
import sys
from tqdm import tqdm

depression_keywords = {
    "lexical": ["бессмысленно", "устал", "надоело", "никому не нужен", "ненавижу себя", "депресс"],
    "isolation": ["одиночество", "все ушли", "остался один", "никого рядом", "изолирован"],
    "self_focus": ["я пустой", "я не могу", "мне плохо", "не вижу смысла"],
    "structure": ["жить. больно.", "нет сил", "нет смысла"],
    "physiology": ["не могу спать", "нет аппетита", "встать с кровати", "не хочу есть"]
}

depression_keywords_eng = {
    "lexical": ["meaningless", "tired", "fed up", "nobody needs me", "hate myself", "depress"],
    "isolation": ["loneliness", "everyone left", "left alone", "nobody around", "isolated"],
    "self_focus": ["i'm empty", "i can't", "i feel bad", "see no point", "no purpose"],
    "structure": ["living hurts", "no strength", "no meaning", "pointless"],
    "physiology": ["can't sleep", "no appetite", "get out of bed", "don't want to eat"]
}

def clean_post_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)  # убираем ссылки
    text = re.sub(r"[!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]", "", text)  # убираем только определенные спецсимволы, сохраняем эмодзи
    text = re.sub(r"\s+", " ", text)  # убираем лишние пробелы
    return text.strip().lower()

# def extract_relevant_communities(communities: list) -> list:
#     return [{"name": comm.get("name", ""), "description": comm.get("description", "")} for comm in communities]

def is_depressive(user_data: dict) -> bool:
    """Определяет, является ли пользователь депрессивным на основе текстов его постов"""
    all_posts_text = " ".join([post["text"] for post in user_data["posts"]])
    
    for _, words in depression_keywords.items():
        for word in words:
            if word.lower() in all_posts_text.lower():
                return True
    
    for _, words in depression_keywords_eng.items():
        for word in words:
            if word.lower() in all_posts_text.lower():
                return True
    
    return False

def clean_user(user: dict) -> dict:
    cleaned_user = {
        "user_id": user.get("id"),
        "sex": user.get("sex"),
        "city": user.get("city", {}).get("title"),
        "followers_count": user.get("followers_count", 0),
        "alcohol": user.get("personal", {}).get("alcohol"),
        "smoking": user.get("personal", {}).get("smoking"),
        "life_main": user.get("personal", {}).get("life_main"),
        "people_main": user.get("personal", {}).get("people_main"),
        "status": clean_post_text(user.get("status", "")),
        "posts": [],
        # "communities": extract_relevant_communities(user.get("communities", []))
    }
    
    for post in user.get("posts", []):
        text = clean_post_text(post.get("text", ""))
        if text:  # пропускаем пустые посты
            cleaned_post = {
                "text": text,
                "date": post.get("date"),
                "likes": post.get("likes", {}).get("count", 0),
                "comments": post.get("comments", {}).get("count", 0),
                "reposts": post.get("reposts", {}).get("count", 0),
                "views": post.get("views", {}).get("count", 0)
            }
            cleaned_user["posts"].append(cleaned_post)
    
    return cleaned_user

def clean_dataset(dataset: list) -> list:
    cleaned_data = []
    for user in dataset:
        cleaned = clean_user(user)
        # оставляем только пользователей с постами
        if cleaned["posts"]:
            cleaned_data.append(cleaned)
    return cleaned_data

def process_directory(input_dir, processed_user_ids, all_cleaned_data):
    if not os.path.exists(input_dir):
        print(f"Предупреждение: директория {input_dir} не существует, пропускаем")
        return 0
    
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"Предупреждение: не найдены JSON файлы в директории {input_dir}")
        return 0
    
    print(f"Найдено {len(json_files)} JSON файлов для обработки в директории {input_dir}")
    
    processed_count = 0
    
    for file_path in tqdm(json_files, desc=f"Обработка файлов в {input_dir}"):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                users = json.load(file)
                
                if not isinstance(users, list):
                    print(f"Ошибка: данные в файле {file_path} не являются списком пользователей")
                    continue
                
                cleaned_users = clean_dataset(users)
                
                unique_users = []
                for user in cleaned_users:
                    user_id = user.get("user_id")
                    if user_id and user_id not in processed_user_ids:
                        processed_user_ids.add(user_id)
                        unique_users.append(user)
                    # elif user_id:
                        # print(f"Найден дубликат пользователя с ID: {user_id}, пропускаем")
                
                all_cleaned_data.extend(unique_users)
                processed_count += len(unique_users)
                
                # print(f"Обработано {len(unique_users)} уникальных пользователей из файла {file_path}")
                
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")
    
    return processed_count

def process_all_files(input_dirs=['data'], output_dir='dataset'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    
    all_cleaned_data = []
    processed_user_ids = set()
    total_processed = 0
    
    print(f"Будут обработаны следующие директории: {', '.join(input_dirs)}")
    
    for input_dir in input_dirs:
        directory_processed = process_directory(input_dir, processed_user_ids, all_cleaned_data)
        total_processed += directory_processed
        # print(f"Всего обработано {directory_processed} уникальных пользователей из директории {input_dir}")
    
    depressive_count = 0
    for user_data in all_cleaned_data:
        is_user_depressive = is_depressive(user_data)
        user_data["label"] = 1 if is_user_depressive else 0
        if is_user_depressive:
            depressive_count += 1
    
    print(f"Анализ депрессивных пользователей завершен: найдено {depressive_count} из {len(all_cleaned_data)} ({depressive_count/len(all_cleaned_data)*100:.2f}%)")
    
    user_id_mapping = {}
    for i, user_data in enumerate(all_cleaned_data):
        original_id = user_data["user_id"]
        user_id_mapping[original_id] = i
        user_data["user_id"] = i
    
    print(f"Анонимизация завершена. IDs заменены на последовательные числа от 0 до {len(all_cleaned_data)-1}")
    
    output_file = os.path.join(output_dir, "data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"Обработка завершена! Всего обработано {len(all_cleaned_data)} уникальных пользователей.")
    print(f"Результаты сохранены в файл {output_file}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_dirs = sys.argv[1:]
        process_all_files(input_dirs)
    else:
        process_all_files(['data'])
