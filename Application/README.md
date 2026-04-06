# CaneNexus - Running the Application

This document provides all necessary commands to run the **CaneNexus** project locally, including the Backend (Flask API), Frontend (Angular UI), and the Test Suite.

---

## 1. Prerequisites

Before starting, ensure you have the following installed on your machine:
- **Python 3.12+**
- **Node.js (v20+)** and **npm**
- **MongoDB** running locally on default port `27017`

### Database Setup
```bash
# Ensure MongoDB is running on localhost:27017
# No manual setup needed — collections are created automatically on first write!
```

---

## 2. Running the Backend

The backend server relies on a Python virtual environment. *(Refer to the `installs.txt` file in the root directory for a full breakdown of the required modules).*

Open your terminal and run the following:

```cmd
:: 1. Activate your virtual environment (from the project root directory)
venv\Scripts\activate

:: 2. Navigate to the backend directory
cd Application\backend

:: 3. Set your Gemini API key
set GEMINI_API_KEY=your_api_key_here

:: 4. Start the Flask server
python app.py
```
> **Success:** The backend server will start on `http://localhost:5000`

---

## 3. Running the Frontend

The frontend is an Angular 20 application. Open a **new terminal tab/window** and run the following:

```cmd
:: 1. Navigate to the frontend directory
cd Application\frontend

:: 2. Install dependencies (only needed the first time)
npm install

:: 3. Start the Angular development server
npm start
```
> **Success:** The frontend UI will open automatically at `http://localhost:4200`

---

## 4. Running the Test Suite (Backend)

The test suite consists of ~84 unit tests that cover inference logic, knowledge graph traversal, API endpoint validation, schema construction, and the Gemini integration.

Everything is mocked under the hood, meaning you do **not** need the database or massive `.pth` AI models to be active in order to run them!

```cmd
:: 1. Turn on your virtual environment (from project root)
venv\Scripts\activate

:: 2. Navigate to the backend directory
cd Application\backend

:: 3. Run the full test suite with verbose output
pytest tests/ -v
```

*To run a specific file, you can pass its path (e.g., `pytest tests/test_steel_kg.py -v`).*
