import os
import streamlit as st
from common.session import reset_app
from common.crawler import prepare_crawler_command, run_crawler, display_results

def step4_crawling():
    """Компонент для четвертого шага - запуск краулинга и отображение результатов"""
    st.subheader("Запуск краулинга")
    
    status_area = st.empty()
    progress_area = st.empty()
    
    try:
        status_area.info("Подготовка к запуску...")
        command, output_file = prepare_crawler_command()
        
        status_area.empty()
        
        returncode = run_crawler(command)
        
        if returncode == 0 and os.path.exists(output_file):
            progress_area.empty()
            display_results(output_file)
        else:
            status_area.error("❌ Не удалось запустить краулер. Проверьте параметры и повторите попытку.")
    
    except Exception as e:
        status_area.error(f"❌ Ошибка при запуске краулера: {e}")
    
    finally:
        if hasattr(st.session_state, 'users_temp_file') and st.session_state.users_temp_file and os.path.exists(st.session_state.users_temp_file):
            try:
                os.unlink(st.session_state.users_temp_file)
            except Exception as e:
                print(f"Ошибка при удалении временного файла: {e}")
    
    # Кнопка для начала заново
    st.button("Начать заново", on_click=reset_app) 