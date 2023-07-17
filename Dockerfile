# Use Miniconda with Python 3.7 as a base image
FROM continuumio/miniconda3:4.7.12

# Avoid prompt during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update --allow-releaseinfo-change && apt-get install -y \
   git \
   libsm6 \
   libxext6 \
   libxrender-dev \
   make \
   wget \
   gcc \
   m4 \
   tar \
   libncurses5-dev \
   libncursesw5-dev \
   libgl1-mesa-glx && \
   apt-get clean && rm -rf /var/lib/apt/lists/*

# Update conda
RUN conda update -y conda

# Install pip
RUN conda install -y pip

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install dependencies inside a Conda environment
RUN conda create -n rougecondaenv python=3.7
SHELL ["conda", "run", "-n", "rougecondaenv", "/bin/bash", "-c"]

# Extract and build autoconf and automake
ADD autoconf-2.69.tar.gz /usr/src/
ADD automake-1.14.tar.gz /usr/src/
RUN cd /usr/src/autoconf-2.69 && \
    ./configure && \
    make && \
    make install && \
    cd ../automake-1.14 && \
    ./configure && \
    make && \
    make install

# Run Make
RUN make
