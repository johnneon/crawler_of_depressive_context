import streamlit as st
from common.session import go_to_step_3

def step2_auth_input():
    """Компонент для второго шага - аутентификации"""
    st.subheader("Шаг 2: Данные для входа в ВКонтакте")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.login = st.text_input(
            "Логин ВКонтакте:", 
            value=st.session_state.login,
            placeholder="email или телефон"
        )
    with col2:
        st.session_state.password = st.text_input(
            "Пароль:", 
            value=st.session_state.password,
            type="password"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("Назад", on_click=lambda: setattr(st.session_state, 'step', 1))
    with col2:
        st.button("Далее", on_click=go_to_step_3) 