#!/bin/bash
# Wait for RabbitMQ and Qdrant to be ready before starting
set -e

echo "Waiting for dependencies to be ready..."

# Wait for RabbitMQ
echo -n "Checking RabbitMQ..."
for i in {1..30}; do
    if timeout 1 bash -c "echo > /dev/tcp/localhost/5672" 2>/dev/null; then
        echo " ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo " FAILED - timeout after 30s"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# Wait for Qdrant
echo -n "Checking Qdrant..."
for i in {1..30}; do
    if timeout 1 bash -c "echo > /dev/tcp/localhost/6334" 2>/dev/null; then
        echo " ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo " FAILED - timeout after 30s"
        exit 1
    fi
    echo -n "."
    sleep 1
done

echo "All dependencies ready! Starting $@"
exec "$@"

