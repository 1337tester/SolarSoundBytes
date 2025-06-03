from fastapi import FastAPI

app = FastAPI()


# Root endpoint
@app.get('/')
def index():
    return {'Hola SolarSoundBytes :) '}


