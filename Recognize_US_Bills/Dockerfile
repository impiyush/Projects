
FROM python:3.8.2-slim-buster

# set work directory
RUN mkdir -p usbills
WORKDIR /usr/src/app/usbills

# set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY requirements.txt /usr/src/app/usbills
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install -r requirements.txt --no-cache-dir \ 
    && apt-get purge -y --auto-remove build-essential

# copy project
COPY . /usr/src/app/usbills

EXPOSE 5000
CMD [ "python" , "app.py"]
