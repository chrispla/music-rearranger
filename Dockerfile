FROM python:3.7.17-bullseye

RUN apt-get update --fix-missing && apt-get install libsndfile1 ffmpeg libsox-fmt-all sox -y

WORKDIR /app

COPY requirements.txt .

RUN apt-get install -y python3-scipy
RUN pip install Cython==0.29.24
RUN pip install numpy==1.19.2
RUN apt-get install -y portaudio19-dev
RUN pip install -r requirements.txt
