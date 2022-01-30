FROM python:3.9-slim

RUN pip3 install --upgrade pip setuptools
RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

VOLUME /data
VOLUME /fetch_data
VOLUME /logs
