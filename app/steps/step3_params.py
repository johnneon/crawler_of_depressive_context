import streamlit as st
from common.session import start_crawling

def step3_parameters():
    """Компонент для третьего шага - дополнительных параметров"""
    st.subheader("Шаг 3: Дополнительные параметры")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.scroll_count = st.slider(
            "Количество прокруток (глубина сбора):", 
            min_value=1, 
            max_value=50, 
            value=st.session_state.scroll_count
        )
    with col2:
        st.session_state.visible_browser = st.checkbox(
            "Показывать браузер во время работы", 
            value=st.session_state.visible_browser
        )
    
    st.session_state.predict_depression = st.checkbox(
        "Анализировать тексты на признаки депрессии", 
        value=st.session_state.predict_depression
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("Назад", on_click=lambda: setattr(st.session_state, 'step', 2))
    with col2:
        st.button("Запустить краулинг", on_click=start_crawling) 