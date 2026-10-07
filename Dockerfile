FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Core (lightweight) deps by default. To bake in the ML stack:
#   docker build --build-arg ML=1 .
ARG ML=0
COPY backend/requirements.txt backend/requirements-ml.txt backend/
RUN if [ "$ML" = "1" ]; then pip install -r backend/requirements-ml.txt; \
    else pip install -r backend/requirements.txt; fi

COPY backend ./backend
# Hosts inject $PORT; default 8000 locally
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
