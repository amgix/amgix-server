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

# Fix permissions for mounted volumes (especially when /data is mounted from host)
# Ensure directories exist and have correct ownership for RabbitMQ
mkdir -p /data/rabbitmq /data/qdrant
chown -R rabbitmq:rabbitmq /data/rabbitmq 
chmod 755 /data/rabbitmq 

# Create named pipe for supervisord output (FIFO doesn't store data, just passes it through)
mkfifo /tmp/supervisord_output

# Start supervisord in background, redirecting output to the pipe
/usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf > /tmp/supervisord_output 2>&1 &
SUPERVISORD_PID=$!

# Tag and output supervisord's messages in background
while IFS= read -r line; do
    if [[ "$line" =~ ^\[.*\] ]]; then
        # Already tagged by child service, pass through
        echo "$line"
    else
        # Supervisord's own message, add tag
        echo "[supervisord] $line"
    fi
done < /tmp/supervisord_output &
LOGGER_PID=$!

# Wait for supervisord to exit
wait $SUPERVISORD_PID

