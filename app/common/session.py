import streamlit as st

def init_session_state():
    """Инициализация состояния приложения"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
        st.session_state.input_method = "Ввести вручную"
        st.session_state.user_input = ""
        st.session_state.user_file = None
        st.session_state.login = ""
        st.session_state.password = ""
        st.session_state.scroll_count = 10
        st.session_state.visible_browser = False
        st.session_state.predict_depression = True
        st.session_state.users_temp_file = None

def reset_app():
    """Сброс состояния приложения к начальным значениям"""
    st.session_state.step = 1
    st.session_state.input_method = "Ввести вручную"
    st.session_state.user_input = ""
    st.session_state.user_file = None
    st.session_state.login = ""
    st.session_state.password = ""
    st.session_state.scroll_count = 10
    st.session_state.visible_browser = False
    st.session_state.predict_depression = True
    st.session_state.users_temp_file = None
    st.rerun()

def go_to_step_2():
    """Проверка и переход ко второму шагу"""
    if st.session_state.input_method == "Ввести вручную" and not st.session_state.user_input:
        st.error("Необходимо ввести хотя бы одного пользователя")
    elif st.session_state.input_method == "Загрузить файл" and not st.session_state.user_file:
        st.error("Необходимо загрузить файл со списком пользователей")
    else:
        st.session_state.step = 2

def go_to_step_3():
    """Проверка и переход к третьему шагу"""
    if not st.session_state.login or not st.session_state.password:
        st.error("Необходимо ввести логин и пароль ВКонтакте")
    else:
        st.session_state.step = 3

def start_crawling():
    """Запуск процесса краулинга (переход к шагу 4)"""
    st.session_state.step = 4 