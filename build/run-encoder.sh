#!/usr/bin/env sh
set -eu

# Read AMGIX_AMQP_URL from environment
if [ -z "${AMGIX_AMQP_URL:-}" ]; then
  echo "AMGIX_AMQP_URL environment variable is not set" >&2
  exit 1
fi

# # Set up LD_LIBRARY_PATH for CUDA libraries (needed for ONNX Runtime CUDA provider)
# # Find all nvidia library directories and add them to LD_LIBRARY_PATH
# if [ -d /usr/local/lib/python3.12/site-packages/nvidia ]; then
#   cuda_lib_paths=$(find /usr/local/lib/python3.12/site-packages/nvidia -type d -name 'lib' 2>/dev/null | tr '\n' ':')
#   if [ -n "${cuda_lib_paths}" ]; then
#     export LD_LIBRARY_PATH="${cuda_lib_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
#   fi
# fi

# Check if NVIDIA GPU is available
echo "Checking if NVIDIA GPU is available..."
set +e
nvidia="$(nvidia-smi -L 2>/dev/null)"
set -e
if [ -z "${nvidia}" ]; then
  export AMGIX_CUDA='false'
  echo "NVIDIA GPU not found"
else
  export AMGIX_CUDA='true'
  echo "NVIDIA GPU found"
fi

# Run the appropriate service based on AMGIX_ENCODER_ROLE
case "${AMGIX_ENCODER_ROLE:-"all"}" in
  "index")
    echo "Starting EncoderService only (index role)..."
    exec python -m src.encoder.encoder --service index
    ;;
  "query")
    echo "Starting RpcService only (query role)..."
    exec python -m src.encoder.encoder --service query
    ;;
  "all")
    echo "Starting both EncoderService and RpcService..."
    exec python -m src.encoder.encoder --service all
    ;;
  *)
    echo "Usage: AMGIX_ENCODER_ROLE=[index|query|all]" >&2
    echo "  index: Run only EncoderService (document indexing)" >&2
    echo "  query: Run only RpcService (search and validation)" >&2
    echo "  all:   Run both services (default)" >&2
    exit 1
    ;;
esac


