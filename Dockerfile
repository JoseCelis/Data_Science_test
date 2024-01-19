FROM PYTHON:3.11-slim

RUN apt update -q -y && apt install -y --no-install-recomends gcc build-essential rsync && apt clean

WORKDIR /app
COPY . /app

ENV LC_ALL C.UTF-8
ENV LANG=C.UTF-8

RUN pip install pipenv
RUN pipenv install --deploy
RUN pipenv run pip freeze

ENTRYPOINT pipenv run python main_script.py