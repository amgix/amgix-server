#!/bin/bash
# Wait for embedded RabbitMQ and Qdrant when URLs match baked-in defaults (see entrypoint.sh)
set -e

echo "Waiting for dependencies to be ready..."

if [ "${AMGIX_AMQP_URL}" = "${AMGIX_DEFAULT_AMQP_URL}" ]; then
    echo -n "Checking RabbitMQ..."
    for i in {1..30}; do
        if timeout 1 bash -c "echo > /dev/tcp/localhost/5672" 2>/dev/null; then
            echo " ready!"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo " FAILED - timeout after 30s"
            exit 1
        fi
        echo -n "."
        sleep 1
    done
else
    echo "Skipping RabbitMQ wait (external AMGIX_AMQP_URL)"
fi

if [ "${AMGIX_DATABASE_URL}" = "${AMGIX_DEFAULT_DATABASE_URL}" ]; then
    echo -n "Checking Qdrant..."
    for i in {1..30}; do
        if timeout 1 bash -c "echo > /dev/tcp/localhost/6334" 2>/dev/null; then
            echo " ready!"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo " FAILED - timeout after 30s"
            exit 1
        fi
        echo -n "."
        sleep 1
    done
else
    echo "Skipping Qdrant wait (external AMGIX_DATABASE_URL)"
fi

echo "All dependencies ready! Starting $@"
exec "$@"
