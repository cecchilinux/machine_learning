FROM ubuntu:17.10


#--------

# ---> Configure Jupyter <---
COPY jupyter_notebook_config.py /root/.jupyter/
# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /
# ---> Open Ports <---
# TensorBoard and Jupiter
EXPOSE 6006 8888

# ENV HOME=/notebooks
# WORKDIR $HOME
WORKDIR "./notebooks"

# Configure a password for access
ENV PASSWORD=ml

ENV CONDA_DIR=/Miniconda3

ENV PATH=/Miniconda3/bin:${PATH}

CMD ["/run_jupyter.sh", "--allow-root"]
#---------


# Install Python 3.6 and some utilities
RUN apt-get update && apt-get upgrade && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        wget \
        python \
        python-dev \
        python3.6 \
        python3.6-dev \
        htop \
        openssl \
        ca-certificates \
        pkg-config \
        software-properties-common \
        libxtst6 \
        libglu1 \
        && apt-get clean \
        && rm -rf /var/lib/apt-get/lists/* /tmp/* /var/tmp/*

# Get pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py

# Install the Google Cloud SDK


# ---> Install Miniconda 3 <---


RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p $CONDA_DIR -b && \
    rm Miniconda3-latest-Linux-x86_64.sh

RUN conda update -y conda && \
    conda update -y --all && \
    conda config --add channels conda-forge && \
    conda create -n deep_learning_lab && \
    /bin/bash -c "source activate deep_learning_lab" && \
    conda install -y \
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
    pytest \
    scikit-image \
    glob2 \
    opencv && \
    pip install --upgrade datalab && \
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl && \
    conda install keras pytorch torchvision -c soumith
