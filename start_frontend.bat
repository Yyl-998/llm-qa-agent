@echo off
cd /d %~dp0
echo 启动 Streamlit 前端 http://localhost:8501
.venv\Scripts\python.exe -m streamlit run frontend/app.py
pause
