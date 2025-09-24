## To build the image

```
docker build -t neural_network:v2 -t neural_network:latest .
```

## Runnig the container, for training

```
docker run --user root -v $(pwd):/app neural_network:latest python modules/defineNetwork.py
```

## Runnig the container, for testing

```
docker run --user root neural_network python module/testing.py
```

##deploymetn in linux

```
sudo apt install python3 python3-pip python3-venv -y
sudo apt install git 
```