# ChatPDF Backend (FastAPI)

FastAPI backend service for the ChatPDF RAG application.

## Prerequisites

- Python 3.8+ (recommended)
- pip
- Virtual environment (recommended)

## Setup

1. **Create and activate virtual environment**
   It is recommended to use Python 3.8 for better compatibility.

```bash
# Create virtual environment (using Python 3.8)
python3.8 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**

Create a `.env` file in the backend directory:
```env
DEEPSEEK_API_KEY=your_api_key_here
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

## Running the Backend

1. **Start the FastAPI server**
```bash
# Basic run
uvicorn main:app --reload --port 8000

# With host binding (to allow external access)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2. **Access the API**
- API Documentation: http://localhost:8000/docs
- OpenAPI Spec: http://localhost:8000/openapi.json

## API Endpoints

- `POST /api/upload` - Upload PDF documents
- `POST /api/chat` - Send chat messages
- `GET /api/sessions` - List chat sessions
- `POST /api/sessions` - Create new session
- `DELETE /api/sessions/{session_id}` - Delete session

## Project Structure
```
backend/
├── app/
│   ├── api/
│   ├── core/
│   └── services/
├── uploads/
├── config.py
├── main.py
└── requirements.txt
```

## Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=app
```

## Common Issues

1. **ModuleNotFoundError**
   - Verify virtual environment is activated
   - Confirm all dependencies are installed

2. **Permission Errors**
   - Ensure write permissions for uploads directory
   - Check file ownership of vector store directory

3. **CORS Issues**
   - Verify CORS_ORIGINS in .env matches frontend URL
   - Check API endpoint URLs in frontend

4. **SQLite Version Error**
    - ChromaDB requires SQLite >= 3.35.0. Follow the instructions in the application's error message to upgrade SQLite on your system.

## Deleting the Virtual Environment

To delete the virtual environment, simply remove the `venv` directory:

```bash
rm -rf venv
```
