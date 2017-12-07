# How to use

### dependences

- docker
- docker-compose

## Build the image (to do only the first time)

- create a folder for yours datasets:

  ```bash
  mkdir ~/datasets
  mkdir ~/scripts
  ```

- open the terminal on the docker folder (where Dockerfile is placed)

- build with command (it can take a lot of time):

  ```bash
  docker build -t ml-image .
  ```



## Run the image

- open the terminal on the folder

- run with command:

  ```bash
  docker-compose up
  ```

- open Jupiter on browser (psw: ml):

  ```URL
  localhost:8888
  ```

- run :

  ```bash
  docker exec -it machine_learning_tf tensorboard --logdir tf_logs/
  ```

- open it on browser (URL):

  ``` URL
  localhost:6006
  ```




## Connect on a new TTY

```bash
docker exec -it [container-id] bash
```
cartella degli scripts
```bash
cd ../scripts
```
