FROM python:3.11-slim

# Системные пакеты (в т.ч. сертификаты и tzdata)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) сначала зависимости — это ускоряет сборки за счёт кеша
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) затем всё остальное
COPY . .

# Запуск бота (unbuffered, чтобы логи сразу печатались)
CMD ["python", "-u", "interior_bot_openrouter.py"]
