from fastapi import FastAPI, Request

app = FastAPI()

# @app.on_event("startup")
# def load_model_once():
#     app.state.model = load_model(stage="Production")


# Root endpoint
@app.get('/')
def index():
    return {'Hola SolarSoundBytes :) '}

# predict endpoint, just for workflow implementation

@app.get('/predict')
def predict(value_1, value_2, request:Request):
    # model = request.app.state.model
    return dict(result = int(value_1)-int(value_2))
