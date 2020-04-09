FROM tensorflow/tensorflow:latest-py3

# set work directory
WORKDIR /usr/src/app

# set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr
# ENV PYTHONUNBUFFERED 1

# copy project and install dependencies
COPY . /usr/src/app
RUN apt-get update \
    && pip install --upgrade pip \
    && pip install -r requirements.txt --no-cache-dir \ 
    && pip install --upgrade tensorflow

EXPOSE 5000
CMD [ "python" , "app.py"]
