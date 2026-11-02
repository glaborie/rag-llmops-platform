# The MLflow server image, with the Postgres driver added.
#
# The upstream image ships MLflow without a PostgreSQL driver, so a server
# that is pointed at a Postgres backend store fails at boot with
# "No module named 'psycopg2'". The driver is all that is missing.
FROM ghcr.io/mlflow/mlflow:v2.16.2

RUN pip install --no-cache-dir psycopg2-binary
