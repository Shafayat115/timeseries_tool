FROM python:3.11-slim

WORKDIR /app

# system deps for scientific python + orbit/cmdstan downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md streamlit_app.py /app/
COPY src /app/src

RUN pip install -U pip && pip install -e ".[app,bsts]"

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
