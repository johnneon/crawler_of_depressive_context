from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import re

DEFAULT_WAIT_TIME = 10

def extract_post_id(post_element):
    """
    Извлекает ID поста из элемента
    Args:
        post_element: Selenium WebElement, представляющий пост
        
    Returns:
        str: числовой ID поста или пустая строка
    """
    try:
        full_id = post_element.get_attribute("data-post-id")
        if full_id and '_' in full_id:
            parts = full_id.split('_')
            if len(parts) >= 2:
                return int(parts[1])
        return full_id if full_id else ""
    except Exception as e:
        print(f"Ошибка при извлечении ID поста: {e}")
        return 0


def extract_post_text(post_element):
    """
    Извлекает текст поста из элемента с data-testid='showmoretext',
    очищая его от всех HTML-тегов и ссылок
    
    Args:
        post_element: Selenium WebElement, представляющий пост
        
    Returns:
        str: очищенный текст поста без HTML-тегов и ссылок
    """
    try:
        showmoretext_element = post_element.find_element(By.CSS_SELECTOR, "div[data-testid='showmoretext']")
        if showmoretext_element:
            text = showmoretext_element.get_attribute('innerHTML')
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'http\S+|www\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#\w+', '', text)
            text = text.lower()
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
    except Exception as e:
        pass
    
    return ""

def extract_posts(driver):
    """
    Извлекает посты со страницы
    
    Args:
        driver: экземпляр Selenium WebDriver
        
    Returns:
        list: список постов с данными
    """
    try:
        print("Извлекаем посты...")
        wait = WebDriverWait(driver, DEFAULT_WAIT_TIME)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div._post")))
        
        post_elements = driver.find_elements(By.CSS_SELECTOR, "div._post")
        print(f"Найдено постов: {len(post_elements)}")
        
        posts = []
        for post in post_elements:
            try:
                post_data = {
                    "text": "",
                    "post_id": 0,
                }

                post_data["post_id"] = extract_post_id(post)
                post_data["text"] = extract_post_text(post)
                
                if post_data["text"] != "":
                    posts.append(post_data)
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