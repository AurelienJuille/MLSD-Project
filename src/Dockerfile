FROM python:3.9-slim
ENV PYTHONUNBUFFERED True

# Set the working directory to /app
WORKDIR /app

COPY requirements.txt /app/

# Install the required packages
RUN pip install -r requirements.txt

COPY app.py /app/
COPY templates /app/templates
COPY static /app/static

# TO REMOVE after adding the riot api
COPY data/processed_dataset.csv /app/data/processed_dataset.csv

# ONLY FOR LOCAL TESTING
COPY lolffate-key.json /app/lolffate-key.json

# Flask environment variables
ENV FLASK_APP=app.py

# Set the port for the Flask application
ENV PORT 5001

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app