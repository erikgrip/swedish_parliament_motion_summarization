FROM python:3.9-buster

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Install Python dependencies
COPY requirements/prod.txt ./requirements.txt
RUN sed -i 's/cu113/cpu/g' requirements.txt
RUN pip install --upgrade pip~=21.0.0
RUN pip install -r requirements.txt

# Copy relevant directories
COPY motion_title_generator/ ./motion_title_generator
COPY utils/ ./utils
COPY api_server/ ./api

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 /repo/api/app.py