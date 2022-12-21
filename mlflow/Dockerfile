FROM python:3.9.13

EXPOSE 5555



RUN pip install mlflow psycopg2==2.8.*

WORKDIR ${SERVER_DIR}

# wait as long as 60 seconds for posgres to init db, which takes a lot of time particularly in mac osx
 CMD  mlflow server \
    --backend-store-uri file:///artifact/mlruns \
    --default-artifact-root /artifact/mlruns \
    --host 0.0.0.0 \
    --port 5555
