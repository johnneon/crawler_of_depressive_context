from selenium.webdriver.common.by import By

def check_profile_availability(driver, target_user):
    """
    Проверяет доступность профиля пользователя ВКонтакте
    
    Args:
        driver: экземпляр Selenium WebDriver
        target_user: ID пользователя для проверки
        
    Returns:
        bool: True если профиль доступен, False если закрыт или не существует
    """
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
        
        return True
            
    except Exception as e:
        print(f"Ошибка при проверке доступности профиля: {e}")
        return False 