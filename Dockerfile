FROM tensorflow/tensorflow:1.15.2-gpu-py3
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev
COPY ./requirements.txt /
RUN pip3 install -r /requirements.txt
COPY . /SelFlow
