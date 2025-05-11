import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
import re

def extract_name(driver):
    """
    Извлекает имя пользователя из профиля
    """
    try:
        name_element = driver.find_element(By.CSS_SELECTOR, "h2.OwnerPageName, .OwnerPageName")
        if name_element:
            try:
                full_text = name_element.text
                
                surname_element = name_element.find_elements(By.CSS_SELECTOR, ".OwnerPageName__noWrapText")
                if surname_element and surname_element[0].text.strip():
                    surname = surname_element[0].text.strip()
                    
                    first_name_parts = []
                    for word in full_text.split():
                        if word not in surname and not any(word.lower().startswith(skip) for skip in ['заходил', 'была', 'online']):
                            first_name_parts.append(word)
                    
                    first_name = ' '.join(first_name_parts)
                    first_name = re.sub(r'[^\w\s]', '', first_name).strip()
                    
                    full_name = f"{first_name} {surname}".strip()
                    return full_name
                else:
                    words = full_text.split()
                    clean_words = []
                    
                    for word in words:
                        if (word.lower().startswith('заходил') or 
                            word.lower().startswith('была') or 
                            word.lower() == 'online' or
                            word == "•" or
                            word.lower() == "мне" or
                            word.lower() == "уже"):
                            continue
                        clean_words.append(word)
                    
                    return ' '.join(clean_words)
            except Exception as inner_e:
                print(f"Ошибка при извлечении имени из новой структуры: {inner_e}")
                # запасной вариант - берем весь текст и чистим его
                name_text = name_element.text.split('\n')[0].strip()
                return ' '.join([part for part in name_text.split() 
                                if not part.lower().startswith('заходил') and 
                                not part.lower().startswith('была') and
                                not part.lower() == 'online'])
        
        name_element = driver.find_element(By.CSS_SELECTOR, "h1[class*='page_name']")
        if name_element:
            return name_element.text.strip()
            
    except Exception as e:
        print(f"Не удалось получить имя пользователя: {e}")
    return ""

def extract_screen_name(driver):
    """
    Извлекает строковый идентификатор пользователя (screen_name) из URL профиля
    """
    try:
        current_url = driver.current_url
        if "vk.com/" in current_url:
            profile_id = current_url.split("vk.com/")[1].split("?")[0]
            return profile_id
    except Exception as e:
        print(f"Не удалось получить screen_name пользователя из URL: {e}")
    return ""

def extract_numeric_id(driver):
    """
    Извлекает числовой ID пользователя из ссылок на стену
    """
    try:
        wall_links = driver.find_elements(By.CSS_SELECTOR, ".ui_tab.ui_tab_new")
        for link in wall_links:
            href = link.get_attribute("href")
            if href and "/wall" in href:
                # извлекаем ID из ссылки типа "/wall123456" или "/wall123456?own=1"
                numeric_id_match = re.search(r'/wall(\d+)', href)
                if numeric_id_match:
                    numeric_id = numeric_id_match.group(1)
                    return int(numeric_id)
    except Exception as e:
        print(f"Не удалось получить числовой ID пользователя из ссылок: {e}")
    return 0

def extract_status(driver):
    """
    Извлекает статус пользователя
    """
    try:
        status_element = driver.find_element(By.CSS_SELECTOR, "div.ProfileInfo__status")
        if status_element:
            span_elements = status_element.find_elements(By.CSS_SELECTOR, "span span")
            if span_elements:
                for span in span_elements:
                    status_text = span.text.strip()
                    if status_text:
                        return status_text
            status_text = status_element.text.strip()
            if status_text:
                return status_text

    except Exception as e:
        print(f"Не удалось получить статус пользователя: {e}")
    
    return ""

def extract_followers_count(driver):
    """
    Извлекает количество подписчиков или друзей
    
    Args:
        driver: экземпляр Selenium WebDriver
        
    Returns:
        int: количество подписчиков/друзей
    """
    try:
        friends_elements = driver.find_elements(By.CSS_SELECTOR, 'a.ProfileGroupHeader span.vkuiHeader__indicator')
        if friends_elements and len(friends_elements) > 0:
            for element in friends_elements:
                count_text = element.text.strip()
                if count_text:
                    return parse_count_text(count_text)
    except Exception:
        pass

    return 0

def parse_count_text(count_text):
    """
    Преобразует текстовое представление количества подписчиков/друзей в число
    
    Args:
        count_text: текстовое представление числа (например, "4,7M", "258", "1,5K")
        
    Returns:
        int: числовое значение
    """
    if not count_text:
        return 0
    
    clean_text = count_text.strip()
    
    try:
        if 'M' in clean_text or 'М' in clean_text:
            multiplier = 1000000
            clean_text = clean_text.replace('M', '').replace('М', '')
        elif 'K' in clean_text or 'К' in clean_text:
            multiplier = 1000
            clean_text = clean_text.replace('K', '').replace('К', '')
        else:
            multiplier = 1
        
        clean_text = clean_text.replace(',', '.')
        value = float(clean_text) * multiplier
        
        return int(value)
    except (ValueError, TypeError) as e:
        print(f"Ошибка при преобразовании числа подписчиков/друзей: {e}")
        return 0

def extract_sex(name=None):
    """
    Извлекает информацию о поле пользователя
    
    Args:
        driver: экземпляр Selenium WebDriver
        name: имя пользователя (если уже было извлечено ранее)
        
    Returns:
        int: 0 - не определено, 1 - женский, 2 - мужской
    """
    # тк в профиле нет никаких признаков пола, то будем пытаться распознать пол по имени
    if not name or name == "":
        return 0
    name_parts = name.split()
    if len(name_parts) >= 2:
        last_name = name_parts[1].lower()
        # если фамилия заканчивается на "а" и не заканчивается на "я" (для фамилий вроде "Митя")
        # то с большой вероятностью это женская фамилия
        if last_name.endswith("а") and not last_name.endswith("я"):
            return 1
        # если фамилия имеет типичные мужские окончания
        elif last_name.endswith("ов") or last_name.endswith("ев") or last_name.endswith("ин") or last_name.endswith("ий"):
            return 2
        
        first_name = name_parts[0].lower()
        
        # женские имена
        female_names = [
            "анна", "мария", "екатерина", "елена", "ольга", "наталья", "татьяна", "ирина", 
            "светлана", "юлия", "марина", "алена", "полина", "виктория", "дарья", "яна", 
            "кристина", "анастасия", "ксения", "евгения", "валерия", "вера", "алина", 
            "инна", "галина", "людмила", "александра", "софья", "софия"
        ]
        
        # мужские имена
        male_names = [
            "александр", "иван", "дмитрий", "михаил", "андрей", "сергей", "владимир", 
            "артем", "никита", "максим", "данил", "егор", "николай", "алексей", "павел", 
            "илья", "кирилл", "антон", "евгений", "даниил", "роман", "виктор", "тимофей",
            "петр", "олег", "игорь", "константин", "артемий", "денис", "глеб"
        ]
        
        if first_name in female_names:
            return 1
        elif first_name in male_names:
            return 2
        
    return 0

def open_details_modal(driver):
    """
    Открывает модальное окно с подробной информацией о профиле
    
    Args:
        driver: экземпляр Selenium WebDriver
        
    Returns:
        function: функция для закрытия модального окна
    """
    try:
        print("Ищем кнопку 'Подробнее'...")
        
        details_button = None
        
        xpath_queries = [
            "//span[contains(@class, 'vkitActionsGroupItem__root') and .//span[contains(text(), 'Подробнее')]]",
            "//span[contains(@class, 'vkitActionsGroupItem__root') and .//span[contains(text(), 'Learn more')]]",
            "//span[contains(@class, 'vkitActionsGroupItem__root')]//svg[contains(@class, 'info_circle_outline_20')]/ancestor::span[contains(@class, 'vkitActionsGroupItem__root')]",
            "//*[contains(text(), 'Подробнее')]/ancestor::span[contains(@class, 'vkitActionsGroupItem__root')]",
            "//*[contains(text(), 'Learn more')]/ancestor::span[contains(@class, 'vkitActionsGroupItem__root')]",
            "//span[contains(text(), 'Подробнее')]",
            "//span[contains(text(), 'Learn more')]",
            "//span[contains(@class, 'ActionsGroupItem')]//span[contains(text(), 'Подробнее')]",
            "//span[contains(@class, 'ActionsGroupItem')]//span[contains(text(), 'Learn more')]",
            "//a[contains(text(), 'Показать подробную информацию')]",
            "//a[contains(text(), 'Show detailed information')]",
            "//*[@data-testid='profile_more_info_button']"
        ]
        
        for xpath in xpath_queries:
            try:
                elements = driver.find_elements(By.XPATH, xpath)
                if elements:
                    details_button = elements[0]
                    break
            except Exception:
                continue
        
        if details_button:
            try:
                details_button.click()
                print("Кнопка 'Подробнее' нажата обычным кликом")
            except Exception as click_err:
                print(f"Не удалось нажать обычным кликом: {click_err}")
                
                try:
                    driver.execute_script("arguments[0].click();", details_button)
                    print("Кнопка 'Подробнее' нажата с помощью JavaScript")
                except Exception as js_err:
                    print(f"Не удалось нажать с помощью JavaScript: {js_err}")
                    
                    try:
                        actions = ActionChains(driver)
                        actions.move_to_element(details_button).click().perform()
                        print("Кнопка 'Подробнее' нажата с помощью ActionChains")
                    except Exception as action_err:
                        print(f"Не удалось нажать с помощью ActionChains: {action_err}")
                        return lambda: None
            
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='modal-close-button']"))
                )
                print("Модальное окно загружено")
                
                def close_modal():
                    try:
                        print("Пытаемся закрыть модальное окно...")
                        
                        close_buttons = []
                        
                        try:
                            close_buttons = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='modal-close-button']")
                        except Exception:
                            pass
                            
                        if not close_buttons:
                            try:
                                close_buttons = driver.find_elements(By.CSS_SELECTOR, "div[class*='ModalDismissButton'] svg[class*='cancel_20']")
                            except Exception:
                                pass
                                
                        if not close_buttons:
                            try:
                                close_xpath = "//*[contains(text(), 'Закрыть') or contains(text(), 'Close')]/ancestor::div[contains(@class, 'Dismiss') or contains(@class, 'Close')]"
                                close_buttons = driver.find_elements(By.XPATH, close_xpath)
                            except Exception:
                                pass
                        
                        if close_buttons:
                            close_button = close_buttons[0]
                            
                            try:
                                close_button.click()
                                print("Модальное окно закрыто обычным кликом")
                            except Exception:
                                try:
                                    driver.execute_script("arguments[0].click();", close_button)
                                    print("Модальное окно закрыто с помощью JavaScript")
                                except Exception:
                                    try:
                                        actions = ActionChains(driver)
                                        actions.move_to_element(close_button).click().perform()
                                        print("Модальное окно закрыто с помощью ActionChains")
                                    except Exception as e:
                                        print(f"Не удалось закрыть модальное окно: {e}")
                                        return False
                            
                            time.sleep(1)
                            return True
                        
                        try:
                            from selenium.webdriver.common.keys import Keys
                            actions = ActionChains(driver)
                            actions.send_keys(Keys.ESCAPE).perform()
                            print("Модальное окно закрыто с помощью клавиши Escape")
                            time.sleep(1)
                            return True
                        except Exception as e:
                            print(f"Не удалось закрыть модальное окно с помощью Escape: {e}")
                            
                        return False
                    except Exception as close_err:
                        print(f"Ошибка при закрытии модального окна: {close_err}")
                        return False
                
                return close_modal
            except TimeoutException:
                print("Таймаут при ожидании появления модального окна")
        else:
            print("Кнопка 'Подробнее' не найдена")
    except Exception as e:
        print(f"Ошибка при открытии модального окна с подробной информацией: {e}")

    return lambda: None


def extract_city(driver):
    """
    Извлекает информацию о городе пользователя из модального окна
    
    Args:
        driver: экземпляр Selenium WebDriver
        
    Returns:
        str: название города или пустая строка, если город не указан
    """
    try:
        city_links = driver.find_elements(
            By.XPATH, 
            "//a[contains(@href, '/search/people?c[name]=0') and contains(@href, 'c[city]')]"
        )
        
        if city_links and len(city_links) > 0:
            for link in city_links:
                city_text = link.text.strip()
                if city_text:
                    return city_text
        
        city_items = driver.find_elements(
            By.XPATH, 
            "//div[contains(@class, 'vkuiModalCard')]//div[contains(text(), 'Город') or contains(text(), 'город') or contains(text(), 'City') or contains(text(), 'city')]/following-sibling::div"
        )
        
        if city_items and len(city_items) > 0:
            for item in city_items:
                city_text = item.text.strip()
                if city_text:
                    return city_text
                    
        alt_city_links = driver.find_elements(
            By.XPATH, 
            "//div[contains(@class, 'vkuiModalCard')]//a[contains(@href, 'city') or contains(@href, 'город')]"
        )
        
        if alt_city_links and len(alt_city_links) > 0:
            for link in alt_city_links:
                city_text = link.text.strip()
                if city_text:
                    return city_text
    except Exception as e:
        print(f"Ошибка при извлечении информации о городе: {e}")
    
    return ""


def extract_alcohol(driver):
    """
    Извлекает информацию об отношении к алкоголю из модального окна
    
    Returns:
        int: 0 - не указано, 1 - резко негативное, 2 - негативное, 3 - компромиссное, 
             4 - нейтральное, 5 - положительное
    """
    try:
        alcohol_elements = driver.find_elements(
            By.XPATH, 
            "//a[contains(@href, 'c[alcohol]')]"
        )
        
        if alcohol_elements:
            for elem in alcohol_elements:
                href = elem.get_attribute("href")
                if href:
                    match = re.search(r'c\[alcohol\]=(\d+)', href)
                    if match:
                        return int(match.group(1))
                    
                text = elem.text.lower()
                if "резко негативное" in text or "very negative" in text:
                    return 1
                elif "негативное" in text or "negative" in text:
                    return 2
                elif "компромиссное" in text or "compromise" in text:
                    return 3
                elif "нейтральное" in text or "neutral" in text:
                    return 4
                elif "положительное" in text or "positive" in text:
                    return 5
        
        rows = driver.find_elements(
            By.XPATH, 
            "//div[contains(@class, 'ProfileModalInfoRow')]"
        )
        
        for row in rows:
            row_text = row.text.lower()
            if "алкоголю" in row_text or "alcohol" in row_text:
                if "резко негативное" in row_text or "very negative" in row_text:
                    return 1
                elif "негативное" in row_text or "negative" in row_text:
                    return 2
                elif "компромиссное" in row_text or "compromise" in row_text:
                    return 3
                elif "нейтральное" in row_text or "neutral" in row_text:
                    return 4
                elif "положительное" in row_text or "positive" in row_text:
                    return 5
    
    except Exception as e:
        print(f"Ошибка при извлечении информации об отношении к алкоголю: {e}")
    
    return 0

def extract_smoking(driver):
    """
    Извлекает информацию об отношении к курению из модального окна
    
    Returns:
        int: 0 - не указано, 1 - резко негативное, 2 - негативное, 3 - компромиссное, 
             4 - нейтральное, 5 - положительное
    """
    try:
        smoking_elements = driver.find_elements(
            By.XPATH, 
            "//a[contains(@href, 'c[smoking]')]"
        )
        
        if smoking_elements:
            for elem in smoking_elements:
                href = elem.get_attribute("href")
                if href:
                    match = re.search(r'c\[smoking\]=(\d+)', href)
                    if match:
                        return int(match.group(1))
                    
                text = elem.text.lower()
                if "резко негативное" in text or "very negative" in text:
                    return 1
                elif "негативное" in text or "negative" in text:
                    return 2
                elif "компромиссное" in text or "compromise" in text:
                    return 3
                elif "нейтральное" in text or "neutral" in text:
                    return 4
                elif "положительное" in text or "positive" in text:
                    return 5
        
        rows = driver.find_elements(
            By.XPATH, 
            "//div[contains(@class, 'ProfileModalInfoRow')]"
        )
        
        for row in rows:
            row_text = row.text.lower()
            if "курению" in row_text or "smoking" in row_text:
                if "резко негативное" in row_text or "very negative" in row_text:
                    return 1
                elif "негативное" in row_text or "negative" in row_text:
                    return 2
                elif "компромиссное" in row_text or "compromise" in row_text:
                    return 3
                elif "нейтральное" in row_text or "neutral" in row_text:
                    return 4
                elif "положительное" in row_text or "positive" in row_text:
                    return 5
    
    except Exception as e:
        print(f"Ошибка при извлечении информации об отношении к курению: {e}")
    
    return 0

def extract_life_main(driver):
    """
    Извлекает информацию о главном в жизни из модального окна
    
    Returns:
        int: 0 - не указано, 1 - семья и дети, 2 - карьера и деньги, 3 - развлечения и отдых, 
             4 - наука и исследования, 5 - совершенствование мира, 6 - саморазвитие, 
             7 - красота и искусство, 8 - слава и влияние
    """
    try:
        priority_elements = driver.find_elements(
            By.XPATH, 
            "//a[contains(@href, 'c[personal_priority]')]"
        )
        
        if priority_elements:
            for elem in priority_elements:
                href = elem.get_attribute("href")
                if href:
                    match = re.search(r'c\[personal_priority\]=(\d+)', href)
                    if match:
                        return int(match.group(1))
                    
                text = elem.text.lower()
                if "семья" in text or "дети" in text or "family" in text or "children" in text:
                    return 1
                elif "карьера" in text or "деньги" in text or "career" in text or "money" in text:
                    return 2
                elif "развлечения" in text or "отдых" in text or "entertainment" in text or "recreation" in text:
                    return 3
                elif "наука" in text or "исследования" in text or "science" in text or "research" in text:
                    return 4
                elif "совершенствование мира" in text or "improving the world" in text:
                    return 5
                elif "саморазвитие" in text or "self-development" in text:
                    return 6
                elif "красота" in text or "искусство" in text or "beauty" in text or "art" in text:
                    return 7
                elif "слава" in text or "влияние" in text or "fame" in text or "influence" in text:
                    return 8
        
        rows = driver.find_elements(
            By.XPATH, 
            "//div[contains(@class, 'ProfileModalInfoRow')]"
        )
        
        for row in rows:
            row_text = row.text.lower()
            if "главное в жизни" in row_text or "life priorities" in row_text or "main in life" in row_text:
                if "семья" in row_text or "дети" in row_text or "family" in row_text or "children" in row_text:
                    return 1
                elif "карьера" in row_text or "деньги" in row_text or "career" in row_text or "money" in row_text:
                    return 2
                elif "развлечения" in row_text or "отдых" in row_text or "entertainment" in row_text or "recreation" in row_text:
                    return 3
                elif "наука" in row_text or "исследования" in row_text or "science" in row_text or "research" in row_text:
                    return 4
                elif "совершенствование мира" in row_text or "improving the world" in row_text:
                    return 5
                elif "саморазвитие" in row_text or "self-development" in row_text:
                    return 6
                elif "красота" in row_text or "искусство" in row_text or "beauty" in row_text or "art" in row_text:
                    return 7
                elif "слава" in row_text or "влияние" in row_text or "fame" in row_text or "influence" in row_text:
                    return 8
    
    except Exception as e:
        print(f"Ошибка при извлечении информации о главном в жизни: {e}")
    
    return 0

def extract_people_main(driver):
    """
    Извлекает информацию о главном в людях из модального окна
    
    Returns:
        int: 0 - не указано, 1 - ум и креативность, 2 - доброта и честность, 3 - красота и здоровье, 
             4 - власть и богатство, 5 - смелость и упорство, 6 - юмор и жизнелюбие
    """
    try:
        priority_elements = driver.find_elements(
            By.XPATH, 
            "//a[contains(@href, 'c[people_priority]')]"
        )
        
        if priority_elements:
            for elem in priority_elements:
                href = elem.get_attribute("href")
                if href:
                    match = re.search(r'c\[people_priority\]=(\d+)', href)
                    if match:
                        return int(match.group(1))
                    
                text = elem.text.lower()
                if "ум" in text or "креативность" in text or "intelligence" in text or "creativity" in text:
                    return 1
                elif "доброта" in text or "честность" in text or "kindness" in text or "honesty" in text:
                    return 2
                elif "красота" in text or "здоровье" in text or "beauty" in text or "health" in text:
                    return 3
                elif "власть" in text or "богатство" in text or "power" in text or "wealth" in text:
                    return 4
                elif "смелость" in text or "упорство" in text or "courage" in text or "persistence" in text:
                    return 5
                elif "юмор" in text or "жизнелюбие" in text or "humor" in text or "zest for life" in text:
                    return 6
        
        rows = driver.find_elements(
            By.XPATH, 
            "//div[contains(@class, 'ProfileModalInfoRow')]"
        )
        
        for row in rows:
            row_text = row.text.lower()
            if "главное в людях" in row_text or "most important in people" in row_text or "important in people" in row_text:
                if "ум" in row_text or "креативность" in row_text or "intelligence" in row_text or "creativity" in row_text:
                    return 1
                elif "доброта" in row_text or "честность" in row_text or "kindness" in row_text or "honesty" in row_text:
                    return 2
                elif "красота" in row_text or "здоровье" in row_text or "beauty" in row_text or "health" in row_text:
                    return 3
                elif "власть" in row_text or "богатство" in row_text or "power" in row_text or "wealth" in row_text:
                    return 4
                elif "смелость" in row_text or "упорство" in row_text or "courage" in row_text or "persistence" in row_text:
                    return 5
                elif "юмор" in row_text or "жизнелюбие" in row_text or "humor" in row_text or "zest for life" in row_text:
                    return 6
    
    except Exception as e:
        print(f"Ошибка при извлечении информации о главном в людях: {e}")
    
    return 0

def set_label():
    # тут будем использовать нейронку для определения метки профиля
    return 0


def extract_profile_info(driver):
    """
    Извлекает основную информацию о профиле пользователя ВКонтакте
    """
    try:
        print("Получаем информацию о профиле...")
        
        profile_data = {
            "user_id": 0,
            "sex": 0,
            "city": "",
            "followers_count": 0,
            "alcohol": 0,
            "smoking": 0,
            "life_main": 0,
            "people_main": 0,
            "status": "",
            "label": 0,

            # дополнительная информация
            "screen_name": "",
            "name": "",
        }
        
        profile_data["name"] = extract_name(driver)
        profile_data["screen_name"] = extract_screen_name(driver)
        profile_data["user_id"] = extract_numeric_id(driver)
        profile_data["status"] = extract_status(driver)
        profile_data["followers_count"] = extract_followers_count(driver)
        
        close_modal_func = open_details_modal(driver)
        
        if close_modal_func:
            try:
                print("Извлекаем дополнительную информацию из модального окна...")
                
                profile_data["sex"] = extract_sex(profile_data["name"])
                profile_data["alcohol"] = extract_alcohol(driver)
                profile_data["smoking"] = extract_smoking(driver)
                profile_data["life_main"] = extract_life_main(driver)
                profile_data["people_main"] = extract_people_main(driver)
                profile_data["city"] = extract_city(driver)
            finally:
                close_modal_func()
        else:
            print("Не удалось открыть модальное окно, пробуем получить информацию с основной страницы...")
            
        profile_data["label"] = set_label()
        
        print(f"Собрана информация о профиле: {profile_data['name']} (id: {profile_data['user_id']})")
        return profile_data
        
    except Exception as e:
        print(f"Ошибка при извлечении информации о профиле: {e}")
        return {}