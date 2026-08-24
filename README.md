# Device Recommender Full-Stack App

This is a RAG-based expert system for Mobile/Laptop recommendations. The project is organized as a monorepo containing both a FastAPI backend (with FAISS indexing and Groq LLM integration) and a Next.js frontend.

## Architecture

- **`/notebook`**: Contains the original data preparation and pipeline notebook, as well as the output Parquet file with generated embeddings.
- **`/backend`**: FastAPI application that provides recommendation endpoints. It reads the pre-computed Parquet file, builds a FAISS index in memory (defaulting to HNSW for low latency), and queries the Groq API for grounded recommendations.
- **`/frontend`**: Next.js App Router application providing a chat-style interface and a device comparison view.
- **`.github/workflows`**: Basic CI pipeline that runs backend tests and frontend build checks.

## How to Run Locally

### 1. Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # Windows: .\venv\Scripts\activate
   # Linux/Mac: source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Set your Groq API key:
   Create a `.env` file in the `/backend` directory:
   ```env
   GROQ_API_KEY=your_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ALLOWED_ORIGINS=http://localhost:3000
   ```
4. Make sure the notebook output file exists at `notebook/output/devices_embeddings.parquet` (run the notebook if it's missing).
5. Start the backend:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### 2. Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Set the backend URL in a `.env.local` file:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```
5. Open [http://localhost:3000](http://localhost:3000) to view the application.

## How to Deploy

### Deploying the Backend (Render / Railway)

**Render:**
1. Connect your GitHub repository to Render.
2. Create a new "Web Service".
3. Render should automatically detect the `render.yaml` file in the root and configure the service (using `backend/Dockerfile` and setting the Docker context to the root directory).
4. **Manual Step**: In the Render dashboard for the new service, add your `GROQ_API_KEY` to the Environment Variables.

**Railway:**
1. Connect your GitHub repository to Railway.
2. Create a new service from the repository.
3. Configure the Root Directory to `/` and Builder to Dockerfile, pointing to `backend/Dockerfile`.
4. Add the `GROQ_API_KEY` to the service variables.

### Deploying the Frontend (Vercel)

1. Import your GitHub repository into Vercel.
2. Set the "Framework Preset" to **Next.js**.
3. Set the "Root Directory" to `frontend`.
4. Add the Environment Variable `NEXT_PUBLIC_API_URL` and set it to the URL of your deployed backend service.
5. Click **Deploy**.

## Testing

Backend tests can be run using `pytest` inside the `backend` directory.
