FROM node:22-bookworm-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VPP_PROJECT_ROOT=/app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 fonts-nanum \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY config/ ./config/
RUN pip install --no-cache-dir -e .
COPY --from=frontend-build /frontend/dist ./frontend/dist
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
