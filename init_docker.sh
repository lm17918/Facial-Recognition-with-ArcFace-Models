# Set variables
IMAGE_NAME="lorenzom_faceid"
TAG="latest"
PATH_TO_DOCKERFILE="./"
CONTAINER_NAME="lorenzom_faceid"

# Build Docker image
docker build -t $IMAGE_NAME:$TAG $PATH_TO_DOCKERFILE

# Run Docker container
docker run -itd -v $(pwd):/workspace/main -v /mnt:/mnt --name $CONTAINER_NAME --gpus all --rm $IMAGE_NAME
