import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

DEFAULT_SCROLL_COUNT = 10

def setup_driver(visible=False):
    """
    Настраивает и инициализирует драйвер Chrome с необходимыми опциями
    
    Args:
        visible (bool): Запуск браузера с видимым GUI (по умолчанию: False - headless режим)
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    
    # опция для работы в headless режиме (без открытия окна браузера)
    if not visible:
        options.add_argument('--headless')
    
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

def scroll_page(driver, scroll_count=DEFAULT_SCROLL_COUNT):
    """
    Прокручивает страницу указанное количество раз
    """
    try:
        print(f"Прокручиваем страницу {scroll_count} раз...")
        for i in range(scroll_count):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
    except Exception as e:
        print(f"Ошибка при прокрутке страницы: {e}") 