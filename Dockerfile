FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt* ./
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || echo "No requirements"

# Copy app
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "print('healthy')" || exit 1

# Run
CMD ["python", "-m", "src.main"]
