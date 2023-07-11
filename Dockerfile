# Use Miniconda with Python 3.7 as a base image
FROM continuumio/miniconda3:4.7.12

# Update and install dependencies.
RUN apt-get update --allow-releaseinfo-change && apt-get install -y \
   git \
   libsm6 \
   libxext6 \
   libxrender-dev \
   make \
   wget \
   gcc \
   libncurses5-dev \
   libncursesw5-dev \
   libgl1-mesa-glx && \
   apt-get clean && rm -rf /var/lib/apt/lists/*


# Update conda
RUN conda update -y conda

# Install pip
RUN conda install -y pip

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory
WORKDIR /app

RUN conda create -n rougecondaenv python=3.7

# Run Make
RUN /bin/bash -c "source activate rougecondaenv && make"