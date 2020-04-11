FROM tensorflow/tensorflow:2.0.0-py3

# set work directory
WORKDIR /usr/src/app

# copy project and install dependencies
COPY . /usr/src/app
RUN apt-get update \
    && pip install --upgrade pip \
    && pip install -r requirements.txt --no-cache-dir

# CMD [ "python" , "app.py"]
