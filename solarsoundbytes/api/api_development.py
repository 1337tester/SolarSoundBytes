from fastapi import FastAPI, Request
from solarsoundbytes.compare_sent_analy.test_sentimental_analysis_calc import create_output_interface
from solarsoundbytes.text_creation.create_text import create_text_from_sent_analy_df


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
def predict(request:Request):
    # model = request.app.state.model
    result_twitter, result_news = create_output_interface()
    result_text = create_text_from_sent_analy_df()
    return result_twitter, result_news, result_text
    # return dict(result = int(value_1)-int(value_2))
