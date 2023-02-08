FROM python:3.6-slim

RUN pip3 install --upgrade pip
RUN pip install --upgrade pip setuptools wheel
RUN apt-get update
RUN apt-get -y install build-essential libpq-dev twine

RUN pip install nose pipenv psycopg2 sklearn coverage

WORKDIR /app

ADD Pipfile /app
RUN pipenv lock -r > requirements.txt \
    && pipenv lock -r --dev > requirements-dev.txt

ADD . /app

RUN pipenv install

CMD "/bin/sh"
