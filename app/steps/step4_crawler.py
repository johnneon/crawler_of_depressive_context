import os
import streamlit as st
from common.session import reset_app
from common.crawler import prepare_crawler_command, run_crawler, display_results

def step4_crawling():
    """Компонент для четвертого шага - запуск краулинга и отображение результатов"""
    st.subheader("Запуск краулинга")
    
    try:
        # Создаем плейсхолдер для вывода
        output_area = st.empty()
        
        # Запускаем краулер
        st.info("Запускаем краулер.")
        command, output_file = prepare_crawler_command()
        returncode = run_crawler(command)
        
        if returncode == 0 and os.path.exists(output_file):
            st.success("Краулинг через subprocess успешно завершен!")
            display_results(output_file)
        else:
            st.error("Не удалось запустить краулер ни одним из способов.")
    
    except Exception as e:
        st.error(f"Ошибка при запуске краулера: {e}")
    
    finally:
        if st.session_state.users_temp_file and os.path.exists(st.session_state.users_temp_file):
            os.unlink(st.session_state.users_temp_file)
    
    st.button("Начать заново", on_click=reset_app) 