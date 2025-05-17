#!/usr/bin/env python
"""
Скрипт для запуска веб-интерфейса краулера депрессивного контекста
Запускает streamlit-приложение из директории app
"""
import os
import sys
import streamlit.web.cli as stcli

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app", "app.py")
    
    sys.argv = ["streamlit", "run", app_path, "--server.headless", "true"]
    sys.exit(stcli.main()) 