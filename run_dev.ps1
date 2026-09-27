# Starts the API (port 8000) and the dashboard (port 3000) in two windows.
# First time only:  python -m venv .venv; .venv\Scripts\pip install -r backend\requirements.txt
#                   (torch CPU wheel: add --extra-index-url https://download.pytorch.org/whl/cpu)
#                   cd frontend; npm install; cd ..
$root = $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; ..\.venv\Scripts\python -m uvicorn app.main:app --port 8000 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"
Start-Sleep -Seconds 8
Start-Process "http://localhost:3000"
