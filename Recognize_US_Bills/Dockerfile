
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
RUN pip install --upgrade pip
COPY requirements.txt /usr/src/app/usbills
RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/app/usbills

EXPOSE 5000
CMD [ "python" , "app.py"]