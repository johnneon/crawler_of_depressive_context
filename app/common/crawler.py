import os
import sys
import time
import json
import tempfile
import streamlit as st
import subprocess

def prepare_crawler_command():
    """Подготовка команды для запуска краулера"""
    if st.session_state.input_method == "Загрузить файл" and st.session_state.user_file:
        users_temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt')
        content = st.session_state.user_file.getvalue().decode('utf-8')
        users_temp_file.write(content)
        users_temp_file.close()
        st.session_state.users_temp_file = users_temp_file.name
        users_arg = f"--users-file {users_temp_file.name}"
    else:
        users_arg = f"--users \"{st.session_state.user_input}\""
    
    output_file = os.path.join(tempfile.gettempdir(), f"vk_data_{int(time.time())}.json")
    
    command = [
        sys.executable,
        "crawler/main.py",
        f"--login {st.session_state.login}",
        f"--password {st.session_state.password}",
        users_arg,
        f"--scrolls {st.session_state.scroll_count}",
        f"--output {output_file}"
    ]
    
    if st.session_state.visible_browser:
        command.append("--visible")
        
    if st.session_state.predict_depression:
        command.append("--predict-depression")
    
    return " ".join(command), output_file

def run_crawler(command):
    """Запуск краулера и отображение процесса"""
    output_area = st.empty()
    
    with st.spinner("Идет сбор данных..."):
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        output_text = ""
        for line in iter(process.stdout.readline, ""):
            output_text += line
            output_area.code(output_text)
        
        process.wait()
    
    return process.returncode

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