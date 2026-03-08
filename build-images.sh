#!/bin/sh

set -e

AMGIX_VERSION="v1.0.0-beta1.1"

# export BUILDKIT_PROGRESS=plain

echo "Building API image..."
docker build -t amgix-api:${AMGIX_VERSION} -f build/Dockerfile-api --build-arg AMGIX_VERSION=${AMGIX_VERSION} .

echo "Building Encoder GPU image..."
docker build -t amgix-encoder:${AMGIX_VERSION}-gpu -f build/Dockerfile-encoder \
  --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
  --build-arg AMGIX_BUILD_FE=GPU \
  --build-arg AMGIX_BUILD_ST=GPU \
  .

echo "Building Encoder CPU image..."
docker build -t amgix-encoder:${AMGIX_VERSION} -f build/Dockerfile-encoder \
  --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
  --build-arg AMGIX_BUILD_FE=CPU \
  --build-arg AMGIX_BUILD_ST=CPU \
  .

# echo "Building Encoder FE GPU image..."
# docker build -t amgix-encoder:${AMGIX_VERSION}-fe-gpu -f build/Dockerfile-encoder-gpu-fe \
#   --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
#   .

# echo "Building Encoder FE CPU image..."
# docker build -t amgix-encoder:${AMGIX_VERSION}-fe -f build/Dockerfile-encoder \
#   --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
#   --build-arg AMGIX_BUILD_FE=CPU \
#   --build-arg AMGIX_BUILD_ST="" \
#   .

# almost the same size as full GPU image
# echo "Building Encoder ST GPU image..."
# docker build -t amgix-encoder:${AMGIX_VERSION}-st-gpu -f build/Dockerfile-encoder \
#   --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
#   --build-arg AMGIX_BUILD_FE="" \
#   --build-arg AMGIX_BUILD_ST=GPU \
#   .

# almost the same size as full CPU image
# echo "Building Encoder ST CPU image..."
# docker build -t amgix-encoder:${AMGIX_VERSION}-st -f build/Dockerfile-encoder \
#   --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
#   --build-arg AMGIX_BUILD_FE="" \
#   --build-arg AMGIX_BUILD_ST=CPU \
#   .

echo "Building Encoder NoEmbed image..."
docker build -t amgix-encoder:${AMGIX_VERSION}-noembed -f build/Dockerfile-encoder \
  --build-arg AMGIX_VERSION=${AMGIX_VERSION} \
  --build-arg AMGIX_BUILD_FE="" \
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