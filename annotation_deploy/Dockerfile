# Lightweight image for the Ulysses-MTRAG human-annotation app (Streamlit).
# The app needs no ML stack — only streamlit + pandas + scikit-learn (for kappa).
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir streamlit pandas scikit-learn

# App + the fixed 200-item sample (both annotators rate the same items).
COPY human_eval_app.py ./
COPY results/human_eval_sample.csv ./results/human_eval_sample.csv

EXPOSE 8501
# Annotations are written to /app/results/human_eval_<name>.csv — mount a volume there.
CMD ["streamlit", "run", "human_eval_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
