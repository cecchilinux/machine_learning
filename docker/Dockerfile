# TODO: Move to 17.10
FROM ubuntu:17.10

LABEL maintainer="Enrico Ceccolini <enrico.ceccolini@mail.com>"

# Install Python 3.6 and some utilities
RUN apt-get update && apt-get upgrade && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        wget \
        git \
        python \
        python-dev \
        python3.6 \
        python3.6-dev \
        unzip \
        tar \
        htop \
        openssl \
        ca-certificates \
        pkg-config \
        software-properties-common \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt-get/lists/*

# Make Python 3.6 the default python
# RUN ln -s /usr/bin/python3.6 /usr/bin/python

# Get pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py

# Install the Google Cloud SDK


# ---> Install Miniconda 3 <---
ENV CONDA_DIR=/Miniconda3

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p $CONDA_DIR -b && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/Miniconda3/bin:${PATH}

RUN conda update -y conda && \
    conda update -y --all

# Setup conda-forge
RUN conda config --add channels conda-forge

# Create a single conda env
RUN conda create -n deep_learning_lab
RUN /bin/bash -c "source activate deep_learning_lab"

# ---> Install our scientific stack <---
RUN conda install -y \
    jupyter \
    jupyterlab \
    jupyter_contrib_nbextensions \
    ipykernel \
    numpy \
    pandas \
    scikit-learn \
    dask \
    altair \
    matplotlib \
    bokeh \
    seaborn \
    flask \
    flake8 \
    autopep8 \
    pytest

# Install google datalab
RUN pip install --upgrade datalab

# ---> Deep Learnig Libraries <---
# Install TensorFlow
RUN pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl
# RUN conda install \
#     tensorflow \
#     tensorboard

# Install Keras
RUN conda install keras

# Install Pytorch
RUN conda install pytorch torchvision -c soumith

# ---> Configure Jupyter <---
# Configure a password for access
ENV PASSWORD=ml
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy sample notebooks.
# COPY notebooks /notebooks

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

# ---> Open Ports <---
# TensorBoard
EXPOSE 6006
# Jupyter
EXPOSE 8888

# ENV HOME=/notebooks
# WORKDIR $HOME
WORKDIR "./notebooks"

# TODO COPY git_resurrect.sh $HOME
# TODO RUN git_resurrect.sh

CMD ["/run_jupyter.sh", "--allow-root"]
