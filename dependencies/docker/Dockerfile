#From nvidia/cuda:12.2.0-base-ubuntu22.04
#From ubuntu:22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update
RUN apt install -y python3-dev python3-pip
RUN echo "alias python=python3" >> ~/.bashrc

RUN mkdir /home/tmp
COPY ./requirements/requirements.txt /home/tmp/
WORKDIR /home/tmp
RUN pip install -r requirements.txt
WORKDIR /
RUN rm -rf /home/tmp
