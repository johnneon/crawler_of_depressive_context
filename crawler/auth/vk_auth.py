import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

DEFAULT_WAIT_TIME = 10

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
        
        # предупреждение на случай если на акке есть 2FA
        print("⚠️ ВНИМАНИЕ: Если появилось окно 'Безопасность Windows' с запросом ключа доступа - ОТМЕНИТЕ его!")
        print("Нажмите 'Отмена' в диалоговом окне Windows и дождитесь продолжения скрипта.")
        
        # даем время закрыть диалоговое окно
        time.sleep(5)
        print("Ожидаем появления элементов на новой странице...")
        
        try:
            print("Ищем кнопку входа...")
            submit_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-test-id='submit_btn']")))
            print("Кнопка входа найдена. Нажимаем...")
            submit_btn.click()
        except Exception as e:
            pass
        
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