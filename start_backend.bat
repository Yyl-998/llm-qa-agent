@echo off
cd /d %~dp0
echo [1/2] 启动 FastAPI 后端 http://localhost:8000
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
