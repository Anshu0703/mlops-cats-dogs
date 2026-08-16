FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer) so Docker caches this step
# and doesn't reinstall torch/etc. every time you change application code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code and trained model
COPY src/ ./src/
COPY models/model.pt ./models/model.pt

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]