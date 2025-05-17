import streamlit as st
from common.session import go_to_step_2

def step1_user_input():
    """Компонент для первого шага - ввода пользователей"""
    st.subheader("Шаг 1: Список пользователей для анализа")
    
    st.session_state.input_method = st.radio(
        "Выберите способ ввода пользователей:",
        ["Ввести вручную", "Загрузить файл"],
        index=0 if st.session_state.input_method == "Ввести вручную" else 1,
        horizontal=True
    )
    
    if st.session_state.input_method == "Ввести вручную":
        st.session_state.user_input = st.text_area(
            "Введите ID пользователей или ссылки на профили (через запятую):",
            value=st.session_state.user_input,
            placeholder="durov, https://vk.com/id1, team"
        )
        st.session_state.user_file = None
    else:
        st.session_state.user_file = st.file_uploader(
            "Загрузите файл со списком пользователей (по одному на строку):",
            type=["txt"]
        )
    
    st.button("Далее", on_click=go_to_step_2) 