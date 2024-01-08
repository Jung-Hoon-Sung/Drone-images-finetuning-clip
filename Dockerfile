# Use the specified CUDA image as the base image
FROM gpuci/cuda:11.5.0-devel-ubuntu20.04

# Set the timezone (optional)
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add NVIDIA GPG key
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Update and install some basic tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file into the container
COPY requirements.txt /tmp/

# Install Python dependencies from requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

COPY . /workspace/

# Set up the working directory
WORKDIR /workspace

# Command to run on container start (for example, run a shell)
CMD [ "/bin/bash" ]
