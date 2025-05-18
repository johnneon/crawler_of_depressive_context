import os
import sys
import time
import json
import streamlit as st
import subprocess

def prepare_crawler_command():
    """Подготовка команды для запуска краулера"""
    temp_dir = os.path.join(os.getcwd(), "app", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    if st.session_state.input_method == "Загрузить файл" and st.session_state.user_file:
        users_file_path = os.path.join(temp_dir, f"users_{int(time.time())}.txt")
        with open(users_file_path, 'w', encoding='utf-8') as f:
            content = st.session_state.user_file.getvalue().decode('utf-8')
            f.write(content)
        st.session_state.users_temp_file = users_file_path
    
    output_file = os.path.join(temp_dir, f"vk_data_{int(time.time())}.json")
    
    command = [
        sys.executable,
        os.path.join(os.getcwd(), "crawler", "main.py")
    ]
    
    if not st.session_state.skip_auth:
        command.extend(["--login", st.session_state.login])
        command.extend(["--password", st.session_state.password])
    else:
        command.append("--skip-auth")
    
    if st.session_state.input_method == "Загрузить файл" and st.session_state.users_temp_file:
        command.extend(["--users-file", st.session_state.users_temp_file])
    else:
        command.extend(["--users", st.session_state.user_input])
    
    command.extend(["--scrolls", str(st.session_state.scroll_count)])
    command.extend(["--output", output_file])
    
    if st.session_state.visible_browser:
        command.append("--visible")
    
    if st.session_state.predict_depression:
        command.append("--predict-depression")
    
    return command, output_file

def run_crawler(command_args):
    """Запуск краулера и отображение процесса"""
    status_container = st.empty()
    
    try:
        my_env = os.environ.copy()
        project_root = os.getcwd()
        my_env['PYTHONPATH'] = project_root
        my_env['PYTHONIOENCODING'] = 'utf-8'
        my_env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
        my_env['WDM_LOG_LEVEL'] = '0'
        my_env['WDM_PRINT_FIRST_LINE'] = 'False'
        my_env['PATH'] = os.path.join(project_root, "app", "temp") + os.pathsep + my_env.get('PATH', '')
        
        status_container.info("🚀 Запуск краулера...")
        time.sleep(2)
        
        process = subprocess.Popen(
            command_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=1,
            env=my_env
        )
        
        messages = [
            "📋 Подготовка списка пользователей...",
            "🔍 Собираем данные профилей...",
            "⏳ Процесс может занимать продолжительное время, просьба проявить терпение...",
            "🧭 Краулер загружает и анализирует страницы пользователей...",
            "📊 Извлечение и анализ постов пользователей...",
            "🔄 Работа продолжается, пожалуйста подождите..."
        ]
        
        show_time = 5
        
        for i in range(len(messages)):
            if process.poll() is not None:
                break
                
            status_container.info(messages[i])
            
            for _ in range(int(show_time * 5)):
                if process.poll() is not None:
                    break
                time.sleep(0.2)
        
        message_index = 0
        while process.poll() is None:
            status_container.info(messages[message_index])
            message_index = (message_index + 1) % len(messages)
            
            for _ in range(int(show_time * 5)):
                if process.poll() is not None:
                    break
                time.sleep(0.2)
                
                try:
                    if process.stdout.peek():
                        process.stdout.readline()
                except (AttributeError, IOError):
                    pass
        
        returncode = process.returncode
        output, _ = process.communicate()
        
        if returncode != 0:
            status_container.error(f"❌ Краулер завершился с ошибкой (код {returncode})")
            
            try:
                error_text = output.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    error_text = output.decode('cp1251')
                except:
                    error_text = "Не удалось декодировать сообщение об ошибке"
            
            error_lines = error_text.strip().split('\n')[-10:]
            st.error("Последние сообщения вывода:")
            for line in error_lines:
                st.code(line)
                
            return returncode
        
        status_container.success("✅ Краулинг успешно завершен!")
        return 0
        
    except Exception as e:
        status_container.error(f"❌ Ошибка при запуске краулера: {str(e)}")
        st.exception(e)
        return 1

def display_results(output_file):
    """Отображение результатов краулинга"""
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        st.subheader("Результаты анализа")
        
        if isinstance(data, list):
            found_depressed = 0
            for i, user_data in enumerate(data):
                is_depressed = False
                if "label" in user_data:
                    is_depressed = user_data.get("label") == 1
                
                prob = 0
                if "probability" in user_data:
                    prob = user_data.get("probability", 0)
                
                icon = "😔" if is_depressed else "🙂"
                if is_depressed:
                    found_depressed += 1
                
                user_name = ""
                if user_data.get('first_name') or user_data.get('last_name'):
                    user_name = f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}"
                elif user_data.get('name'):
                    user_name = user_data.get('name')
                elif user_data.get('user_id'):
                    user_name = f"ID: {user_data.get('user_id')}"
                else:
                    user_name = f"Пользователь {i+1}"
                
                with st.expander(f"{icon} {user_name} ({i+1}/{len(data)})", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        user_id = user_data.get('user_id', user_data.get('id', ''))
                        profile_url = f"https://vk.com/id{user_id}"
                        st.markdown(f"**Пользователь:** [{user_name}]({profile_url})")

                        normalized_prob = 0
                        if prob <= -10:
                            normalized_prob = 0
                        elif prob >= 10:
                            normalized_prob = 100
                        else:
                            normalized_prob = (prob + 10) * 5
                            
                        if normalized_prob > 70:
                            bar_color = "#FF0000"
                        elif normalized_prob > 50:
                            bar_color = "#FFA500"
                        else:
                            bar_color = "#008000"
                            
                        st.markdown(
                            f"""
                            <div style="display:flex; align-items:center; gap:10px;">
                                <p style="white-space:nowrap;"><b>Вероятность депрессии:</b></p>
                                <div style="width:100%; background-color:#f0f0f0; height:20px; border-radius:3px;">
                                    <div style="width:{normalized_prob}%; background-color:{bar_color}; height:20px; border-radius:3px;"></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        depression_text = "Да ✅" if is_depressed else "Нет ❌"
                        st.markdown(f"**Признаки депрессии:** {depression_text}")
                        
                        st.markdown(f"**Вероятность:** {prob:.2f}")
                    
                    if len(user_data.get('posts', [])) > 0:
                        st.markdown(f"**Проанализировано постов:** {len(user_data.get('posts', []))}")
            
            if found_depressed > 0 and len(data) > 0:
                st.success(f"Всего обработано: {len(data)} пользователей. С признаками депрессии: {found_depressed} ({found_depressed/len(data)*100:.1f}%)")
            else:
                st.info(f"Всего обработано: {len(data)} пользователей. Пользователей с признаками депрессии не обнаружено.")
            
        else:
            st.warning("Данные не в ожидаемом формате. Отображаем как JSON:")
            st.json(data)
        
        with open(output_file, 'r') as f:
            st.download_button(
                label="Скачать результаты (JSON)",
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
                file_path = os.path.join(temp_dir, file)
                file_age = time.time() - os.path.getmtime(file_path)
                if file_age > 3600:  # 3600 секунд = 1 час
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Не удалось удалить файл {file_path}: {e}")
    except Exception as e:
        print(f"Ошибка при очистке временных файлов: {e}")
