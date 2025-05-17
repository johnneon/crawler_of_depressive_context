import os
import sys
import streamlit as st
from pathlib import Path

# Добавляем корневой каталог проекта в sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from steps.step1_users import step1_user_input
from steps.step2_auth import step2_auth_input
from steps.step3_params import step3_parameters
from steps.step4_crawler import step4_crawling
from common.session import init_session_state
from common.ui import setup_page, show_project_info

def main():
    """Основная функция приложения"""
    init_session_state()
    setup_page()
    show_project_info()
    
    if st.session_state.step == 1:
        step1_user_input()
    elif st.session_state.step == 2:
        step2_auth_input()
    elif st.session_state.step == 3:
        step3_parameters()
    elif st.session_state.step == 4:
        step4_crawling()

if __name__ == "__main__":
    main() 