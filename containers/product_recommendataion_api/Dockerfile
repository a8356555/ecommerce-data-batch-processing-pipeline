# Copyright 2021 Binflux Inc.
#
# This image is for AI PGT api

FROM python:3.7

LABEL version="1.0.0"

WORKDIR /api
COPY .env .

RUN pip install -U pip
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

COPY ./api .

# Expose ports
EXPOSE 8000

CMD [ "uvicorn","--host=0.0.0.0", "main:app"]
