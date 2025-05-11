import json
import os

DEFAULT_OUTPUT_FILE = 'vk_data.json'

def save_multiple_profiles(profiles_data, filename=DEFAULT_OUTPUT_FILE):
    """
    Сохраняет информацию о нескольких профилях в один JSON-файл в виде массива.
    
    Args:
        profiles_data: список словарей с информацией о профилях и их постах
        filename: имя файла для сохранения
        
    Returns:
        bool: True если данные успешно сохранены, False в случае ошибки
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
        
        profiles_count = len(profiles_data)
        posts_count = sum(len(profile.get('posts', [])) for profile in profiles_data)
        
        print(f"Готово! Сохранено {profiles_count} профилей с {posts_count} постами в {filename}")
        print(f"Общая статистика:")
        
        return True
    except Exception as e:
        print(f"Ошибка при сохранении данных профилей: {e}")
        return False 