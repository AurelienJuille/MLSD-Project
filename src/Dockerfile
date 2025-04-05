FROM python:3.9-slim
ENV PYTHONUNBUFFERED True

# Set the working directory to /app
WORKDIR /app

COPY requirements.txt /app/

# Install the required packages
RUN pip install -r requirements.txt

COPY . /app

# Expose port 8080 for Flask
#EXPOSE 8080

# Flask environment variables
ENV FLASK_APP=app.py

# Set the port for the Flask application
ENV PORT 5001

# Run the Flask application
# CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app