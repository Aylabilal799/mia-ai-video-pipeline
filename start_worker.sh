#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
celery -A tasks worker --loglevel=info
