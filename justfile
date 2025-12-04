CONTAINER_NAME := "mdn-proofread-db"
VOLUME_NAME := "mdn-proofread-db-data"
IMAGE := "mongo:8"
PORT := "27017"

# Start MongoDB container with persistent volume
db-start:
    docker volume create {{VOLUME_NAME}}
    docker run -d --name {{CONTAINER_NAME}} \
        -p {{PORT}}:27017 \
        -v {{VOLUME_NAME}}:/data/db \
        {{IMAGE}}

# Stop and remove the MongoDB container
db-stop:
    -docker stop {{CONTAINER_NAME}}
    -docker rm {{CONTAINER_NAME}}

# Tail MongoDB logs
db-logs:
    docker logs -f {{CONTAINER_NAME}}

# Show MongoDB container status
db-status:
    docker ps -a --filter "name={{CONTAINER_NAME}}"

