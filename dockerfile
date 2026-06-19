FROM python:3.10-slim

WORKDIR /app

# Dependencias do Cassandra
# Instala as dependências de compilação do Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

COPY requirements.txt .

# Força a remoção do pacote conflitante antes de instalar o arquivo de requisitos correto
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y cassandra && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]