FROM nvcr.io/nvidia/tensorflow:22.01-tf2-py3

WORKDIR /app

# Install python3.7 and replace python3.8
RUN apt update && apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.7 python3-pip  python3.7-distutils python3.7-dev
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/bin/python3

# Install system dependencies
RUN apt-get update --fix-missing && apt-get install -y libsndfile1 ffmpeg libsox-fmt-all sox python3-scipy portaudio19-dev

# Install Python dependencies
RUN pip install Cython==0.29.24
RUN pip install numpy==1.19.2
COPY requirements.txt .
RUN pip install -r requirements.txt
