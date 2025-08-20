## To build the image

```
docker build -t neural_network:v2 -t neural_network:latest .
```

What this does is creating two different images, one is the one that will contain the version and the other is an update in the image with the tag 'latest'. This so i can save versions while always having an updated image.

## Runnig the container, for training

```
docker run --user root -v $(pwd):/app neural_network:latest python modules/defineNetwork.py
```

#### `docker run`
- **Purpose**: Creates and starts a new container from an image
- **Creates**: New container instance each time

#### `--user root`
- **Purpose**: Overrides the USER instruction in Dockerfile
- **Effect**: Runs container processes as root user
- **Why needed**: Bypasses permission issues
- **Security trade-off**: Less secure but simpler for development

#### `-v $(pwd):/app`
This is a **volume mount** with two parts:

##### `$(pwd)` - Host Path
- **`$(pwd)`**: Shell command substitution
- **`pwd`**: "Print Working Directory" - shows current directory path
- **Result**: Absolute path of your current directory
- **Example**: If you're in `/home/user/projects/my-pytorch-project`, that's what gets mounted

##### `:/app` - Container Path
- **Destination**: `/app` directory inside the container
- **Effect**: Makes host directory accessible inside container
- **Bidirectional**: Changes in either location reflect in both

## Runnig the container, for testing

```
docker run --user root neural_network python module/testing.py
```