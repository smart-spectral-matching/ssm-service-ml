FROM code.ornl.gov:4567/rse/datastreams/ssm/backend/ssm-ml/python:3.6.13-slim-buster

ARG CI_PROJECT_ID
ENV CI_PROJECT_ID=$CI_PROJECT_ID
ARG TWINE_PASSWORD
ENV TWINE_PASSWORD=$TWINE_PASSWORD
ARG TWINE_USERNAME
ENV TWINE_USERNAME=$TWINE_USERNAME

RUN pip3 install --upgrade pip
RUN pip install --upgrade pip setuptools wheel
RUN apt-get update
RUN apt-get -y install build-essential libpq-dev twine

RUN pip install nose pipenv
RUN pipenv install

ADD . ./

RUN pipenv lock -r > requirements.txt
RUN pipenv lock -r --dev > requirements-dev.txt
RUN pipenv install
RUN pip install psycopg2 sklearn
RUN pipenv run python setup.py sdist
RUN pipenv run python setup.py bdist_wheel
RUN pipenv run nosetests
RUN echo "$TWINE_PASSWORD"
RUN echo "$TWINE_USERNAME"
RUN twine upload --repository-url https://code.ornl.gov/api/v4/projects/${CI_PROJECT_ID}/packages/pypi --verbose dist/*

CMD "/bin/sh"