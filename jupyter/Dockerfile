FROM python:3.9.13

RUN apt-get update && apt-get install -y gcc netcat curl g++ wget pkg-config python3-dev apt-transport-https build-essential ca-certificates curl && apt-get install -y apt-utils nodejs npm


ENV TZ=Asia/Bangkok
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


WORKDIR /code
COPY . /code/

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi
