FROM python:3.10-slim

ARG RUN_ID=unknown
ENV RUN_ID=${RUN_ID}

WORKDIR /app

CMD ["sh", "-c", "echo Downloading model for RUN_ID=$RUN_ID && echo Model ready"]
