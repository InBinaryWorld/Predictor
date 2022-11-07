FROM python:3.6-slim

ENV GIT_DISCOVERY_ACROSS_FILESYSTEM=1
RUN apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y git

ADD requirements.txt /root/req/requirements.txt
WORKDIR /root/req
RUN pip install -r requirements.txt
