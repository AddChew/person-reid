FROM python:3.9-slim-buster

WORKDIR /app

COPY ./serving_requirements.txt /app/serving_requirements

RUN pip install --no-cache--dir --upgrade -r /app/serving_requirements

COPY ./models /app/models

COPY ./outputs /app/outputs

COPY ./src app/src

CMD ["fastapi", "run", "src/api.py", "--port", "4000"]