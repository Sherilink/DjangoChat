#!/usr/bin/env bash
set -o errexit  # exit on error

pip install -r requirements.txt

python chat/manage.py collectstatic --noinput
python chat/manage.py migrate --noinput

gunicorn chat.wsgi:application --chdir chat --bind 0.0.0.0:$PORT

