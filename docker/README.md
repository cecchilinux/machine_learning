# How to use

## dependences

- docker
- docker-compose

## 1. Build the image (to do only the first time)

- create a folder for yours datasets:

  ```bash
  mkdir ~/datasets
  mkdir ~/scripts
  ```

- open the terminal on the docker folder (where Dockerfile is placed)

- build with command (it can take a lot of time):

  ```bash
  docker build --no-cache=true -t ml-image .
  ```

## 2. Run the image, Jupiter automatically starts

- open the terminal on the docker folder (were docker-compose is placed)

- run the container with the command:

  ```bash
  docker-compose up
  ```

- open Jupiter on browser (psw: ml):

  ```URL
  localhost:8888
  ```
## 3. Connect on a new TTY (usefull to run some scripts outside Jupiter)

  ```bash
  docker exec -it [container-id] bash
  ```
  scripts folder
  ```bash
  cd ../scripts
  ```

## Tensorboard
- If you need to use tensorboard run :

  ```bash
  docker exec -it machine_learning_tf tensorboard --logdir tf_logs/
  ```

- open it on browser (URL):

  ``` URL
  localhost:6006
  ```

  ## Clean up
  - see all the images:
  ```bash
  docker images -a
  ```

  - when you modify the Dockerfile is good to clean the dangling images:
  ```bash
  docker images --filter "dangling=true"
  docker rmi $(docker images -q --filter "dangling=true")
  ```

## other commands

```bash
docker build --no-cache=true -t ml-image-gpu -f Dockerfile.gpu .

docker-compose -f docker-compose.yml up
```

```bash
nvidia-docker run \
--rm \
--device /dev/nvidia0:/dev/nvidia0 \
--device /dev/nvidiactl:/dev/nvidiactl \
--device /dev/nvidia-uvm:/dev/nvidia-uvm \
-p 8888:8888 \
-v ~/ML_fileLocali/notebooks:/notebooks/samples \
-v ~/ML_fileLocali/datasets:/datasets \
-v ~/git-repo/mine-repo/machine_learning/notebooks:/notebooks \
-v ~/git-repo/mine-repo/machine_learning/scripts:/scripts \
ml-final-8-6
```
