from fastapi import FastAPI

app = FastAPI()


# Root endpoint
@app.get('/')
def index():
    return {'Hola SolarSoundBytes :) '}

# predict endpoint, just for workflow implementation

@app.get('/predict')
def predict(value_1, value_2):
    return dict(result = int(value_1)-int(value_2))
