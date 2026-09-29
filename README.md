# ParseLegal
Deterministic legal RAG engine querying rental contracts with zero hallucinations and exact clause citations.

## Architecture Update
The application has been updated to use a **React** frontend and a **FastAPI** backend, replacing the legacy Streamlit interface.

## Prerequisites
- Node.js (v18+)
- Python (v3.10+)

## Setup

### Backend (FastAPI)
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Set up your `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will run on `http://localhost:8000`.

### Frontend (React + Vite)
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will run on `http://localhost:5173`.
