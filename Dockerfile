# syntax=docker/dockerfile:1
# =============================================================================
# Verum — single-image build.
# Stage 1 builds the React frontend. Stage 2 runs the FastAPI backend AND
# serves the built frontend from the same origin, so one deploy = one URL.
# =============================================================================

# ---- Stage 1: build the React frontend --------------------------------------
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# Empty VITE_API_URL => the SPA calls the API on its own origin (same service).
ENV VITE_API_URL=""
RUN npm run build        # outputs /fe/dist

# ---- Stage 2: backend + bundled static frontend -----------------------------
FROM python:3.11-slim AS backend
WORKDIR /app

# Install Python deps first (better layer caching).
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code, then the compiled frontend into ./static (main.py serves it).
COPY backend/app ./app
COPY --from=frontend /fe/dist ./static

# Hosts like Render/Railway inject $PORT; default to 8000 locally.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
