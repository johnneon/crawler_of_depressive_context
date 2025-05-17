import os
import sys
import time
import json
import tempfile
import streamlit as st
import subprocess

def prepare_crawler_command():
    """Подготовка команды для запуска краулера"""
    # Создаем директорию для временных файлов внутри приложения
    temp_dir = os.path.join(os.getcwd(), "app", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Временный файл для списка пользователей
    if st.session_state.input_method == "Загрузить файл" and st.session_state.user_file:
        users_file_path = os.path.join(temp_dir, f"users_{int(time.time())}.txt")
        with open(users_file_path, 'w', encoding='utf-8') as f:
            content = st.session_state.user_file.getvalue().decode('utf-8')
            f.write(content)
        st.session_state.users_temp_file = users_file_path
    
    # Путь для выходного файла с результатами
    output_file = os.path.join(temp_dir, f"vk_data_{int(time.time())}.json")
    
    # Создаем прямую команду для запуска краулера
    command = [
        sys.executable,
        os.path.join(os.getcwd(), "crawler", "main.py")
    ]
    
    # Добавляем аргументы для запуска crawler
    if not st.session_state.skip_auth:
        command.extend(["--login", st.session_state.login])
        command.extend(["--password", st.session_state.password])
    else:
        # Если пропускаем авторизацию, добавляем флаг
        command.append("--skip-auth")
    
    # Добавляем пользователей
    if st.session_state.input_method == "Загрузить файл" and st.session_state.users_temp_file:
        command.extend(["--users-file", st.session_state.users_temp_file])
    else:
        command.extend(["--users", st.session_state.user_input])
    
    # Добавляем остальные параметры
    command.extend(["--scrolls", str(st.session_state.scroll_count)])
    command.extend(["--output", output_file])
    
    if st.session_state.visible_browser:
        command.append("--visible")
    
    if st.session_state.predict_depression:
        command.append("--predict-depression")
    
    return command, output_file

def run_crawler(command_args):
    """Запуск краулера и отображение процесса"""
    output_area = st.empty()
    
    my_env = os.environ.copy()
    project_root = os.getcwd()
    my_env['PYTHONPATH'] = project_root
    # Установим кодировку для корректной работы с русскими символами в Windows
    my_env['PYTHONIOENCODING'] = 'utf-8'
    my_env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
    
    # Добавляем переменные для Selenium и Chrome
    my_env['WDM_LOG_LEVEL'] = '0'  # Отключаем логи webdriver-manager
    my_env['WDM_PRINT_FIRST_LINE'] = 'False'
    
    # Указываем явно, где искать chromedriver
    chrome_driver_path = os.path.join(project_root, "app", "temp", "chromedriver.exe")
    my_env['PATH'] = os.path.join(project_root, "app", "temp") + os.pathsep + my_env.get('PATH', '')
    
    with st.spinner("Идет сбор данных..."):
        try:
            # Для отладки - показываем команду
            st.info(f"Выполняем команду: {' '.join(command_args)}")
            
            # Проверяем, есть ли chromedriver, если нет - пробуем скачать
            if not os.path.exists(chrome_driver_path):
                st.info("Локальный chromedriver не найден, пробуем скачать...")
                try:
                    from crawler.utils.browser import download_chromedriver
                    driver_path = download_chromedriver(chrome_driver_path)
                    if driver_path and os.path.exists(driver_path):
                        st.success(f"ChromeDriver успешно загружен в {driver_path}")
                    else:
                        st.warning("Не удалось автоматически загрузить chromedriver. Попробуем запустить без него.")
                except Exception as e:
                    st.warning(f"Ошибка при попытке загрузить chromedriver: {e}")
            
            # Запускаем процесс
            process = subprocess.Popen(
                command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,  # Изменено на False для работы с байтами
                bufsize=1,
                env=my_env
            )
            
            output_text = ""
            for line in iter(process.stdout.readline, b""):
                try:
                    decoded_line = line.decode('utf-8')
                    output_text += decoded_line
                    output_area.code(output_text)
                except UnicodeDecodeError:
                    # Пробуем другие кодировки, если utf-8 не сработал
                    try:
                        decoded_line = line.decode('cp1251')
                        output_text += decoded_line
                        output_area.code(output_text)
                    except:
                        # Если не удалось декодировать, просто пропускаем строку
                        continue
            
            process.wait()
            
            if process.returncode != 0:
                st.error(f"Краулер завершился с ошибкой (код {process.returncode})")
                return 1
                
            return process.returncode
        except Exception as e:
            st.error(f"Ошибка при запуске краулера: {str(e)}")
            return 1

def display_results(output_file):
    """Отображение результатов краулинга"""
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        st.subheader("Результаты анализа")
        st.json(data)
        
        with open(output_file, 'r') as f:
            st.download_button(
                label="Скачать результаты",
                data=f,
                file_name="vk_data.json",
                mime="application/json"
            )
    except Exception as e:
        st.error(f"Ошибка при загрузке результатов: {e}")

def cleanup_temp_files():
    """Функция для очистки временных файлов"""
    try:
        temp_dir = os.path.join(os.getcwd(), "app", "temp")
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                # Не удаляем файлы, созданные меньше 1 часа назад
                file_path = os.path.join(temp_dir, file)
                file_age = time.time() - os.path.getmtime(file_path)
                if file_age > 3600:  # 3600 секунд = 1 час
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Не удалось удалить файл {file_path}: {e}")
    except Exception as e:
        print(f"Ошибка при очистке временных файлов: {e}")
