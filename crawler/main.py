import os
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from crawler.extractors import extract_posts, extract_profile_info
from crawler.auth import login_to_vk
from crawler.utils import setup_driver, scroll_page, save_multiple_profiles, check_profile_availability

DEFAULT_SCROLL_COUNT = 10
DEFAULT_WAIT_TIME = 10 
DEFAULT_OUTPUT_FILE = 'vk_data.json'

def extract_vk_user_id(url_or_id):
    """
    Извлекает ID пользователя ВКонтакте из URL или строки ID
    
    Args:
        url_or_id: URL профиля или ID пользователя
        
    Returns:
        str: ID пользователя
    """
    if url_or_id.startswith('http'):
        match = re.search(r'vk\.com/([^/?]+)', url_or_id)
        if match:
            return match.group(1)
        return url_or_id
    
    return url_or_id

def scrape_vk_user(driver, target_user, scroll_count=DEFAULT_SCROLL_COUNT):
    """
    Собирает данные профиля и постов отдельного пользователя
    
    Args:
        driver: экземпляр Selenium WebDriver
        target_user: ID пользователя или URL профиля
        scroll_count: количество прокруток страницы
        
    Returns:
        tuple: (profile_info, posts) или (None, None) в случае ошибки
    """
    try:
        user_id = extract_vk_user_id(target_user)
        
        profile_url = f"https://vk.com/{user_id}"
        print(f"Переходим на {profile_url}...")
        driver.get(profile_url)
        
        WebDriverWait(driver, DEFAULT_WAIT_TIME).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )
        
        if not check_profile_availability(driver, user_id):
            return None, None
        
        print("Извлекаем информацию о профиле...")
        profile_info = extract_profile_info(driver)
        
        scroll_page(driver, scroll_count)
        
        posts = extract_posts(driver)
        
        if not posts:
            print(f"Не найдено постов на странице пользователя {user_id}.")
            print("Возможно, профиль закрыт или не содержит публичных постов.")
            return profile_info, []
        
        return profile_info, posts
            
    except Exception as e:
        print(f"Ошибка при сборе данных пользователя {target_user}: {e}")
        return None, None

def parse_users_list(users_string):
    """
    Парсит строку с ID пользователей или URL профилей, разделенных запятыми
    
    Args:
        users_string: строка с ID пользователей или URL профилей
        
    Returns:
        list: список ID пользователей или URL профилей
    """
    if not users_string:
        return []
        
    users = [user.strip() for user in users_string.split(',') if user.strip()]
    print(f"Получен список из {len(users)} пользователей из командной строки")
    return users

def load_users_list(file_path):
    """
    Загружает список пользователей из текстового файла
    Поддерживает как ID пользователей, так и URL профилей
    
    Args:
        file_path: путь к файлу со списком пользователей
        
    Returns:
        list: список ID пользователей или URL профилей
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            users = [line.strip() for line in f if line.strip()]
        print(f"Загружен список из {len(users)} пользователей из файла {file_path}")
        return users
    except Exception as e:
        print(f"Ошибка при загрузке списка пользователей: {e}")
        return []

def scrape_vk_posts(login, password, target_users, scroll_count=DEFAULT_SCROLL_COUNT, output_file=DEFAULT_OUTPUT_FILE, visible=False):
    """
    Собирает данные профилей и постов для списка пользователей
    
    Args:
        login: логин ВКонтакте
        password: пароль ВКонтакте
        target_users: список ID пользователей или URL профилей
        scroll_count: количество прокруток страницы
        output_file: путь к файлу для сохранения всех данных
        visible: запуск браузера с видимым GUI (по умолчанию: False - headless режим)
        
    Returns:
        bool: True если хотя бы один профиль был успешно обработан, False в случае ошибки
    """
    driver = None
    success_count = 0
    all_profiles = []
    
    try:
        driver = setup_driver(visible=visible)
        
        if not login_to_vk(driver, login, password):
            return False
        
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for idx, user in enumerate(target_users):
            print(f"\n[{idx+1}/{len(target_users)}] Обрабатываем пользователя: {user}")
            
            profile_info, posts = scrape_vk_user(driver, user, scroll_count)
            
            if profile_info is not None:
                profile_data = profile_info.copy()
                profile_data["posts"] = posts
                
                all_profiles.append(profile_data)
                success_count += 1
                print(f"Данные пользователя {user} успешно получены")
        
        if all_profiles:
            if save_multiple_profiles(all_profiles, output_file):
                print(f"\nВсе данные успешно сохранены в файл {output_file}")
        
        print(f"\nОбработка завершена. Успешно обработано профилей: {success_count}/{len(target_users)}")
        return success_count > 0
    
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        return False
    finally:
        if driver:
            driver.quit()
            print("Браузер закрыт")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Скрипт для сбора постов с ВКонтакте')
    parser.add_argument('--login', type=str, required=True,
                       help='Логин пользователя VK')
    parser.add_argument('--password', type=str, required=True,
                       help='Пароль пользователя VK')
    parser.add_argument('--users', type=str,
                        help='Список ID пользователей или URL профилей через запятую (например: durov,https://vk.com/id1)')
    parser.add_argument('--users-file', type=str,
                        help='Путь к файлу со списком ID пользователей или URL профилей (по одному на строку)')
    parser.add_argument('--scrolls', type=int, default=DEFAULT_SCROLL_COUNT, 
                        help=f'Количество прокруток страницы (по умолчанию: {DEFAULT_SCROLL_COUNT})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f'Путь к файлу для сохранения данных (по умолчанию: {DEFAULT_OUTPUT_FILE})')
    parser.add_argument('--visible', action='store_true',
                        help='Запуск браузера с видимым GUI (по умолчанию: False - headless режим)')
    
    args = parser.parse_args()
    
    target_users = []
    if args.users_file:
        target_users = load_users_list(args.users_file)
    elif args.users:
        target_users = parse_users_list(args.users)
    
    if not target_users:
        print("Не указаны пользователи для обработки. Необходимо указать --users или --users-file.")
        print("Пример: --users durov,https://vk.com/id1")
        exit(1)
    
    success = scrape_vk_posts(args.login, args.password, target_users, args.scrolls, args.output, args.visible)
    
    if success:
        print("Краулинг завершен успешно!")
    else:
        print("Краулинг завершен с ошибками.")
