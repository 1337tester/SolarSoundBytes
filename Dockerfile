FROM python:3.10.6-buster

COPY . . 

RUN pip install --upgrade pip
RUN pip install -e .
RUN pip install -r requirements.txt

# ENV PYTHONPATH=/app

# CMD uvicorn solarsoundbytes.api.api_development:app --host 0.0.0.0 --port $PORT


CMD ["sh", "-c", "streamlit run website/home.py --server.address 0.0.0.0 --server.port $PORT"]




