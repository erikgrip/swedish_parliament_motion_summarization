FROM jupyter/datascience-notebook

RUN pip3 install --upgrade pip setuptools
RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

VOLUME /data
VOLUME /notebooks
