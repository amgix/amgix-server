#!/bin/sh

set -e

cd "$(dirname "$0")/.."

AMGIX_VERSION="v1.0.0-beta1.1"

# export BUILDKIT_PROGRESS=plain

echo "Building API image..."
docker build -t amgix-api:${AMGIX_VERSION} -f build/Dockerfile-api --build-arg AMGIX_VERSION=${AMGIX_VERSION} .

echo "Building Encoder GPU image..."
docker build -t amgix-encoder:${AMGIX_VERSION}-gpu -f build/Dockerfile-encoder \
  --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
  --build-arg AMGIX_BUILD_ST=GPU \
  .

echo "Building Encoder CPU image..."
docker build -t amgix-encoder:${AMGIX_VERSION} -f build/Dockerfile-encoder \
  --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
  --build-arg AMGIX_BUILD_ST=CPU \
  .

echo "Building Encoder NoEmbed image..."
docker build -t amgix-encoder:${AMGIX_VERSION}-noembed -f build/Dockerfile-encoder \
  --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
  --build-arg AMGIX_BUILD_ST="" \
  .

echo "Building Amgix-One CPU image..."
docker build -t amgix-one:${AMGIX_VERSION} -f build/one/Dockerfile-one \
  --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
  --build-arg AMGIX_BUILD_ST=CPU \
  .

echo "Building Amgix-One GPU image..."
docker build -t amgix-one:${AMGIX_VERSION}-gpu -f build/one/Dockerfile-one \
  --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
  --build-arg AMGIX_BUILD_ST=GPU \
  .

echo "Building Amgix-One NoEmbed image..."
docker build -t amgix-one:${AMGIX_VERSION}-noembed -f build/one/Dockerfile-one \
  --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
  --build-arg AMGIX_BUILD_ST="" \
  .

echo "Done"
