FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY requirements.txt /app

RUN pip3 install -r requirements.txt
