#!/bin/bash
set -e

# Function to gracefully shutdown supervisord
shutdown() {
    echo "[entrypoint] Received shutdown signal, shutting down supervisord..."
    if [ -n "$SUPERVISORD_PID" ] && kill -0 "$SUPERVISORD_PID" 2>/dev/null; then
        # Send SIGTERM to supervisord, which will gracefully shut down all children
        kill -TERM "$SUPERVISORD_PID"
        # Wait for supervisord to finish shutting down
        wait $SUPERVISORD_PID 2>/dev/null || true
    fi
    exit 0
}

# Trap SIGTERM and SIGINT to gracefully shutdown
trap shutdown SIGTERM SIGINT

if [ "${AMGIX_DATABASE_URL}" != "${AMGIX_DEFAULT_DATABASE_URL}" ]; then
    echo "[entrypoint] AMGIX_DATABASE_URL != AMGIX_DEFAULT_DATABASE_URL — disabling embedded Qdrant supervisord program"
    sed -i '/^\[program:qdrant\]/,/^\[program:api\]/ s/^autostart=true$/autostart=false/' /etc/supervisor/conf.d/amgix-one.conf
else
    mkdir -p /data/qdrant
fi

if [ "${AMGIX_AMQP_URL}" != "${AMGIX_DEFAULT_AMQP_URL}" ]; then
    echo "[entrypoint] AMGIX_AMQP_URL != AMGIX_DEFAULT_AMQP_URL — disabling embedded RabbitMQ supervisord program"
    sed -i '/^\[program:rabbitmq\]/,/^\[program:qdrant\]/ s/^autostart=true$/autostart=false/' /etc/supervisor/conf.d/amgix-one.conf
else
    # Fix permissions for mounted volumes (especially when /data is mounted from host)
    # Ensure directory exists and have correct ownership for RabbitMQ
    mkdir -p /data/rabbitmq
    chown -R rabbitmq:rabbitmq /data/rabbitmq
    chmod 755 /data/rabbitmq
fi

if [ "${AMGIX_ONE_ENCODER_MODELS}" == "2" ]; then
    echo "[entrypoint] AMGIX_ONE_ENCODER_MODELS=2 — enabling AMGIX_LOAD_MODELS on encoder1"
    sed -i '/^\[program:encoder1\]/,/^\[program:encoder2\]/ s/AMGIX_LOAD_MODELS=false/AMGIX_LOAD_MODELS=true/' /etc/supervisor/conf.d/amgix-one.conf
fi

# Increase the maximum number of open files limit to 65536
ulimit -n 65536

/usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf > >(
    while IFS= read -r line; do
        if [[ "$line" =~ ^\[.*\] ]]; then
            echo "$line"
        else
            echo "[supervisord] $line"
        fi
    done
) 2>&1 &
SUPERVISORD_PID=$!

wait $SUPERVISORD_PID

