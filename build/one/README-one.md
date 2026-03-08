# Amgix-One: All-in-One Container

Single-container deployment of Amgix with all dependencies bundled.

## What's Inside

- **Python 3.12** runtime
- **RabbitMQ** message queue (from official apt repository)
- **Qdrant** vector database (binary v1.15.5)
- **Amgix API** service
- **Amgix Encoder** service (FastEmbed + SentenceTransformers, CPU-only)
- **Supervisord** process manager

## Build

```bash
docker build -f build/Dockerfile-one -t amgix-one:latest .
```

## Run

```bash
docker run -d \
  -p 8234:8234 \
  -v amgix-data:/data \
  --name amgix-one \
  amgix-one:latest
```

## Architecture

All services run inside a single container, communicating via localhost:
- **RabbitMQ**: Port 5672 (internal only)
- **Qdrant**: Port 6334 (internal only)
- **API**: Port 8234 (exposed)
- **Encoder**: No ports (worker)

Data is persisted to `/data` volume:
- `/data/qdrant` - Qdrant storage
- `/data/rabbitmq` - RabbitMQ mnesia

## Logs

All services log to Docker stdout/stderr:
```bash
docker logs -f amgix-one
```

## Use Cases

- **Quick demos** - One command to run
- **Development** - Simple local setup
- **Small deployments** - Single machine, low traffic
- **Edge deployments** - Resource-constrained environments

## Limitations

- CPU-only (no GPU support)
- Single encoder instance (limited parallelism)
- Not horizontally scalable
- Larger image size (~1.6GB vs ~800MB for multi-container)
- All services share container resources

## For Production

Use the full multi-container deployment with docker-compose for:
- GPU support
- Multiple encoder instances
- Better resource isolation
- Horizontal scaling
- Separate service management

