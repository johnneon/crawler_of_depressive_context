import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
import os

DEFAULT_TARGET_USER = 'durov'
DEFAULT_SCROLL_COUNT = 10
DEFAULT_WAIT_TIME = 10 
DEFAULT_OUTPUT_FILE = 'vk_posts.json'

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    # опция для работы в headless режиме (без открытия окна браузера)
    # options.add_argument('--headless')
    
    # отключаем запрос ключа доступа Windows
    options.add_argument('--disable-webauthn-extension')
    options.add_argument('--disable-extensions')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--disable-blink-features=AutomationControlled')
    
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        # обманываем сайт, что это не автоматизированный браузер
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        print(f"Ошибка при инициализации браузера: {e}")
        raise

def login_to_vk(driver, login, password):
    try:
        print("Открываем vk.com...")
        driver.get("https://vk.com")
        
        wait = WebDriverWait(driver, DEFAULT_WAIT_TIME)
        
        print("Нажимаем на кнопку входа...")
        enter_another_way_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='enter-another-way']")))
        enter_another_way_btn.click()
        
        print("Вводим логин...")
        phone_input = wait.until(EC.presence_of_element_located((By.NAME, "login")))
        phone_input.clear()
        phone_input.send_keys(login)
        phone_input.send_keys(Keys.RETURN)
        
        print("Ожидаем перенаправления на id.vk.ru...")
        wait.until(lambda d: "id.vk.ru" in d.current_url)
        
        time.sleep(2)
        
        # Предупреждение на случай если на акке есть 2FA
        print("⚠️ ВНИМАНИЕ: Если появилось окно 'Безопасность Windows' с запросом ключа доступа - ОТМЕНИТЕ его!")
        print("Нажмите 'Отмена' в диалоговом окне Windows и дождитесь продолжения скрипта.")
        
        # Даем время закрыть диалоговое окно
        time.sleep(5)
        print("Ожидаем появления элементов на новой странице...")
        
        try:
            print("Ищем кнопку входа...")
            submit_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-test-id='submit_btn']")))
            print("Кнопка входа найдена. Нажимаем...")
            submit_btn.click()
        except Exception as e:
            print(f"Не удалось найти кнопку входа: {e}")
        
        try:
            print("Ищем кнопку аутентификации...")
            anotherWayLoginBtn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-test-id='anotherWayLogin']")))
            print("Кнопка аутентификации найдена. Нажимаем...")
            anotherWayLoginBtn.click()
        except Exception as e:
            print(f"Не удалось найти кнопку аутентификации: {e}")
            
        time.sleep(1)
            
        try:
            print("Ищем кнопку SMS аутентификации...")
            verificationMethodSmsBtn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-test-id='verificationMethod_sms']")))
            print("Кнопка SMS аутентификации найдена. Нажимаем...")
            verificationMethodSmsBtn.click()
        except Exception as e:
            print(f"Не удалось найти кнопку SMS аутентификации: {e}")
        
        print("Ожидаем отправки СМС кода...")
        try:
            wait.until(EC.presence_of_element_located((By.NAME, "otp-cell")))
            
            sms_code = input("Введите код из СМС (обычно 6 цифр): ")
            sms_code = sms_code.strip()
            otp_cells = driver.find_elements(By.NAME, "otp-cell")
            
            if len(otp_cells) < len(sms_code):
                print(f"Внимание: Код содержит {len(sms_code)} цифр, но найдено только {len(otp_cells)} полей для ввода")
                sms_code = sms_code[:len(otp_cells)]
                
            print("Вводим код из СМС...")
            for i, digit in enumerate(sms_code):
                if i < len(otp_cells):
                    otp_cells[i].clear()
                    otp_cells[i].send_keys(digit)
                    time.sleep(0.2)
                    
            print("После ввода СМС-кода ожидаем поле для ввода пароля...")
            try:
                time.sleep(3)
                
                password_input = wait.until(EC.presence_of_element_located((By.NAME, "password")))
                print("Найдено поле для ввода пароля. Вводим пароль...")
                password_input.clear()
                password_input.send_keys(password)
                password_input.send_keys(Keys.RETURN)
                print("Пароль отправлен")
            except Exception as pass_e:
                print(f"Не удалось найти поле для ввода пароля после СМС: {pass_e}")
                print("Возможно, пароль не требуется или процесс авторизации изменился")
                
        except TimeoutException:
            print("Поля для ввода SMS-кода не найдены. Возможно, используется другой метод аутентификации.")
        except Exception as e:
            print(f"Ошибка при вводе SMS-кода: {e}")
        
        print("Ожидаем завершения авторизации и редиректа на vk.ru/feed...")
        auth_wait = WebDriverWait(driver, DEFAULT_WAIT_TIME * 2)
        
        try:
            auth_wait.until(lambda driver: "vk.ru/feed" in driver.current_url)
            print("Авторизация успешна! Перенаправлены на ленту новостей.")
            return True
        except TimeoutException:
            print("Ожидание редиректа на vk.ru/feed превышено, проверяем другие признаки авторизации...")
            try:
                if "feed" in driver.current_url or "login" not in driver.current_url:
                    print("URL указывает на авторизованное состояние.")
                    return True
                
                if driver.find_elements(By.ID, "top_profile_link") or driver.find_elements(By.ID, "l_pr"):
                    print("Обнаружены элементы авторизованного пользователя. Предполагаем, что авторизация успешна.")
                    return True
            except:
                pass
            
            print("Не удалось подтвердить успешную авторизацию.")
            return False
    except TimeoutException as te:
        print(f"Превышено время ожидания при авторизации: {te}")
        return False
    except Exception as e:
        print(f"Ошибка при авторизации: {e}")
        return False

def scroll_page(driver, scroll_count=DEFAULT_SCROLL_COUNT):
    try:
        print(f"Прокручиваем страницу {scroll_count} раз...")
        for i in range(scroll_count):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            print(f"Прокрутка {i+1}/{scroll_count} выполнена")
    except Exception as e:
        print(f"Ошибка при прокрутке страницы: {e}")

def extract_profile_info(driver):
    """
    Извлекает основную информацию о профиле пользователя ВКонтакте
    """
    try:
        print("Получаем информацию о профиле...")
        profile_data = {}
        
        try:
            name_element = driver.find_element(By.CSS_SELECTOR, "h1[class*='page_name']")
            if name_element:
                profile_data["name"] = name_element.text.strip()
        except Exception as e:
            print(f"Не удалось получить имя пользователя: {e}")
        
        try:
            current_url = driver.current_url
            if "vk.com/" in current_url:
                profile_id = current_url.split("vk.com/")[1].split("?")[0]
                profile_data["user_id"] = profile_id
        except Exception as e:
            print(f"Не удалось получить ID пользователя: {e}")
        
        try:
            status_element = driver.find_element(By.CSS_SELECTOR, "div[class*='profile_info'] span[class*='current_text']")
            if status_element:
                profile_data["status"] = status_element.text.strip()
        except Exception as e:
            pass  # У многих пользователей нет статуса
        
        try:
            followers_element = driver.find_element(By.CSS_SELECTOR, 
                                                 "a[href*='followers'] [class*='header_count']")
            if followers_element:
                followers_text = followers_element.text.strip().replace(" ", "").replace("K", "000")
                profile_data["followers_count"] = int(followers_text) if followers_text.isdigit() else 0
        except Exception as e:
            pass  # У многих пользователей скрыта информация о подписчиках
        
        try:
            info_blocks = driver.find_elements(By.CSS_SELECTOR, "div[class*='profile_info_block']")
            for block in info_blocks:
                block_title_element = block.find_element(By.CSS_SELECTOR, "[class*='profile_info_header']") if block.find_elements(By.CSS_SELECTOR, "[class*='profile_info_header']") else None
                if not block_title_element:
                    continue
                
                block_title = block_title_element.text.strip().lower()
                
                if "город" in block_title:
                    city_element = block.find_element(By.CSS_SELECTOR, "a[class*='profile_info_link']")
                    if city_element:
                        profile_data["city"] = city_element.text.strip()
                
                elif "день рождения" in block_title:
                    bday_element = block.find_element(By.CSS_SELECTOR, "[class*='profile_info_block_content']")
                    if bday_element:
                        profile_data["birthday"] = bday_element.text.strip()
                
                elif "образование" in block_title:
                    edu_element = block.find_element(By.CSS_SELECTOR, "[class*='profile_info_block_content']")
                    if edu_element:
                        profile_data["education"] = edu_element.text.strip()
        
        except Exception as e:
            print(f"Ошибка при получении дополнительной информации о профиле: {e}")
        
        print(f"Информация о профиле собрана: {profile_data}")
        return profile_data
        
    except Exception as e:
        print(f"Ошибка при извлечении информации о профиле: {e}")
        return {}

def extract_posts(driver):
    try:
        print("Извлекаем посты...")
        wait = WebDriverWait(driver, DEFAULT_WAIT_TIME)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div._post")))
        
        post_elements = driver.find_elements(By.CSS_SELECTOR, "div._post")
        print(f"Найдено постов: {len(post_elements)}")
        
        posts = []
        for post in post_elements:
            try:
                post_data = {}
                
                post_id = post.get_attribute("data-post-id")
                if post_id:
                    post_data["post_id"] = post_id
                
                text_element = post.find_element(By.CSS_SELECTOR, "div[class*='vkitPostText__root']") if post.find_elements(By.CSS_SELECTOR, "div[class*='vkitPostText__root']") else None
                if text_element:
                    post_data["text"] = text_element.text.strip()
                else:
                    text_element = post.find_element(By.CSS_SELECTOR, "div.wall_post_text") if post.find_elements(By.CSS_SELECTOR, "div.wall_post_text") else None
                    post_data["text"] = text_element.text.strip() if text_element else ""
                
                date_element = post.find_element(By.CSS_SELECTOR, "a[data-testid='post_date_block_preview']") if post.find_elements(By.CSS_SELECTOR, "a[data-testid='post_date_block_preview']") else None
                if date_element:
                    post_data["date_text"] = date_element.text.strip()
                    post_data["date_href"] = date_element.get_attribute("href")
                    # Можно извлечь timestamp из атрибута если нужно
                
                likes_element = post.find_element(By.CSS_SELECTOR, "div[class*='PostButtonReactions__title']") if post.find_elements(By.CSS_SELECTOR, "div[class*='PostButtonReactions__title']") else None
                if likes_element:
                    likes_text = likes_element.text.strip().replace(" ", "")
                    post_data["likes"] = int(likes_text) if likes_text.isdigit() else 0
                
                reposts_element = post.find_element(By.CSS_SELECTOR, "div.PostBottomAction.share .PostBottomAction__count") if post.find_elements(By.CSS_SELECTOR, "div.PostBottomAction.share .PostBottomAction__count") else None
                if reposts_element:
                    reposts_text = reposts_element.text.strip().replace(" ", "")
                    post_data["reposts"] = int(reposts_text) if reposts_text.isdigit() else 0
                
                comments_element = post.find_element(By.CSS_SELECTOR, "span[class*='comment_count']") if post.find_elements(By.CSS_SELECTOR, "span[class*='comment_count']") else None
                if comments_element:
                    comments_text = comments_element.text.strip().replace(" ", "")
                    post_data["comments"] = int(comments_text) if comments_text.isdigit() else 0
                else:
                    post_data["comments"] = 0  # Если комментарии отключены или их нет
                
                views_element = post.find_element(By.CSS_SELECTOR, "div[class*='like_views']") if post.find_elements(By.CSS_SELECTOR, "div[class*='like_views']") else None
                if views_element:
                    views_text = views_element.text.strip().replace(" ", "").replace("K", "000")  # Обработка формата "1.2K"
                    post_data["views"] = int(views_text) if views_text.isdigit() else 0
                

                posts.append(post_data)
                print(f"Извлечены данные поста {post_id if post_id else 'без ID'}")
            except Exception as e:
                print(f"Ошибка при извлечении данных поста: {e}")
                continue
        
        print(f"Успешно извлечено постов: {len(posts)}")
        return posts
    except TimeoutException:
        print("Не удалось найти посты на странице")
        return []
    except Exception as e:
        print(f"Ошибка при извлечении постов: {e}")
        return []

def save_posts(posts, filename=DEFAULT_OUTPUT_FILE):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        
        print(f"Готово! Посты сохранены в {filename}")
        print(f"Всего сохранено постов: {len(posts)}")
        
        if posts:
            likes_total = sum(post.get('likes', 0) for post in posts)
            reposts_total = sum(post.get('reposts', 0) for post in posts)
            comments_total = sum(post.get('comments', 0) for post in posts)
            
            print(f"Общая статистика:")
            print(f"- Всего лайков: {likes_total}")
            print(f"- Всего репостов: {reposts_total}")
            print(f"- Всего комментариев: {comments_total}")
        
        return True
    except Exception as e:
        print(f"Ошибка при сохранении постов: {e}")
        return False

def scrape_vk_posts(login, password, target_user=DEFAULT_TARGET_USER, scroll_count=DEFAULT_SCROLL_COUNT, output_file=DEFAULT_OUTPUT_FILE):
    driver = None
    try:
        driver = setup_driver()
        
        if not login_to_vk(driver, login, password):
            return False
        
        profile_url = f"https://vk.com/{target_user}"
        print(f"Переходим на {profile_url}...")
        driver.get(profile_url)
        
        WebDriverWait(driver, DEFAULT_WAIT_TIME).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )
        
        try:
            private_elements = driver.find_elements(By.XPATH, 
                "//h3[contains(text(), 'Страница пользователя заблокирована') or contains(text(), 'Страница доступна только авторизованным пользователям')]")
            
            closed_profile_elements = driver.find_elements(By.CSS_SELECTOR, 
                "h3[class*='ClosedProfileBlock__title']")
            
            closed_title_elements = driver.find_elements(By.XPATH,
                "//h3[contains(@class, 'ClosedProfileBlock__title')]")
            
            wall_container = driver.find_elements(By.CSS_SELECTOR, "#page_wall_posts")
            
            if private_elements or closed_profile_elements or closed_title_elements or (len(wall_container) == 0):
                print(f"Ошибка: Профиль пользователя {target_user} закрыт или не существует.")
                print("Невозможно получить доступ к постам этого пользователя.")
                
                if closed_profile_elements:
                    print(f"Обнаружен элемент закрытого профиля по классу: {len(closed_profile_elements)}")
                if closed_title_elements:
                    print(f"Обнаружен заголовок 'Это закрытый профиль': {len(closed_title_elements)}")
                if private_elements:
                    print(f"Обнаружено сообщение о закрытой странице: {len(private_elements)}")
                if len(wall_container) == 0:
                    print("Стена с постами отсутствует")
                    
                return False
                
        except Exception as e:
            print(f"Ошибка при проверке доступности профиля: {e}")
        
        print("Извлекаем информацию о профиле...")
        profile_info = extract_profile_info(driver)
        
        scroll_page(driver, scroll_count)
        
        posts = extract_posts(driver)
        
        if not posts:
            print(f"Не найдено постов на странице пользователя {target_user}.")
            print("Возможно, профиль закрыт или не содержит публичных постов.")
            return False
        
        for post in posts:
            post['profile_info'] = profile_info
            
        return save_posts(posts, output_file)
    
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
    parser.add_argument('--user', type=str, default=DEFAULT_TARGET_USER, 
                        help=f'ID пользователя (по умолчанию: {DEFAULT_TARGET_USER})')
    parser.add_argument('--scrolls', type=int, default=DEFAULT_SCROLL_COUNT, 
                        help=f'Количество прокруток (по умолчанию: {DEFAULT_SCROLL_COUNT})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE, 
                        help=f'Путь к выходному файлу (по умолчанию: {DEFAULT_OUTPUT_FILE})')
    
    args = parser.parse_args()
    
    success = scrape_vk_posts(args.login, args.password, args.user, args.scrolls, args.output)
    
    if success:
        print("Скрапинг завершен успешно!")
    else:
        print("Скрапинг завершен с ошибками.")
