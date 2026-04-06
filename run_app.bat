@echo off
echo ========================================================
echo Starting CaneNexus Development Environment
echo ========================================================
echo.

:: Get the directory of this batch file
set PROJECT_DIR=%~dp0

echo Starting Backend Server...
:: Opens a new terminal, navigates to project root, activates venv, goes to backend, and runs python
start "CaneNexus Backend" cmd /k "cd /d "%PROJECT_DIR%" && call venv\Scripts\activate && cd Application\backend && python app.py"

echo Starting Frontend Server...
:: Opens a new terminal, navigates to project root, goes to frontend, and runs npm start
start "CaneNexus Frontend" cmd /k "cd /d "%PROJECT_DIR%" && cd Application\frontend && npm start"

echo.
echo Both servers have been launched in separate terminal windows!
echo.
echo - Backend should be available at http://localhost:5000
echo - Frontend should be available at http://localhost:4200
echo.
echo You can safely close this launcher window.
pause
