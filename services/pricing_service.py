# from models.pricing_lstm.model import load_model, predict_price

# # Load model once at service startup
# model = load_model()

# def get_price_prediction(request):
#     # Preprocess and predict
#     price = predict_price(model, request)
#     return {"price": price}

def get_price_prediction():
    return {"message":"This is Price Prediction"}