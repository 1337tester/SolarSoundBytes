FROM python:3.10.6-buster

COPY . . 

# Create the streamlit config directories and copy secrets
RUN mkdir -p /root/.streamlit
RUN mkdir -p /.streamlit  
COPY .streamlit/secrets.toml /root/.streamlit/secrets.toml
COPY .streamlit/config.toml /root/.streamlit/config.toml
COPY .streamlit/secrets.toml /.streamlit/secrets.toml
COPY .streamlit/config.toml /.streamlit/config.toml

RUN pip install --upgrade pip
RUN pip install -e .
RUN pip install -r requirements.txt

# expose port for deployment to access by railway.com
EXPOSE 8080

# enable running locally using streamlit
CMD ["sh", "-c", "streamlit run website/home.py --server.address 0.0.0.0 --server.port $PORT"]

