#!/usr/bin/env bash
set -e

# Render provides $PORT
streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-8501}" \
  --server.headless true
