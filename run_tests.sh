#!/usr/bin/env bash
set -e

echo "Running tests..."

# Check if services are running
echo "Checking if API is accessible..."
if ! curl -f http://localhost:8234/health > /dev/null 2>&1; then
    echo "ERROR: API is not accessible at http://localhost:8234"
    echo "Make sure to run: docker compose up --build"
    exit 1
fi

echo "API is accessible. Running tests..."

# Install test dependencies
pip install -r tests/requirements.txt

# Run the tests
pytest tests/ -v

echo "Tests completed!"
