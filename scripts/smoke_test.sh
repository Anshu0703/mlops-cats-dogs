#!/bin/bash
# Post-deploy smoke test: verifies the running service responds correctly.
# Exits non-zero (failing the pipeline) if either check fails.
set -e

HOST="http://localhost:8000"
MAX_RETRIES=10
RETRY_DELAY=3

echo "Waiting for service health check..."
for i in $(seq 1 $MAX_RETRIES); do
  if curl -sf "$HOST/health" > /dev/null; then
    echo "Health check passed."
    break
  fi
  if [ "$i" -eq "$MAX_RETRIES" ]; then
    echo "ERROR: Health check failed after $MAX_RETRIES attempts."
    exit 1
  fi
  sleep $RETRY_DELAY
done

echo "Testing /predict endpoint..."
RESPONSE=$(curl -sf -X POST "$HOST/predict" -F "file=@tests/sample.jpg")

if echo "$RESPONSE" | grep -q "predicted_class"; then
  echo "Smoke test passed. Response: $RESPONSE"
  exit 0
else
  echo "ERROR: Smoke test failed. Response: $RESPONSE"
  exit 1
fi