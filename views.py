from django.shortcuts import render
from .predict import predict_mental_health

def home(request):
    return render(request, 'home.html')  # No need for 'predictor/home.html'

def predict(request):
    prediction = None
    if request.method == "POST":
        user_text = request.POST.get('text')
        prediction = predict_mental_health(user_text)
    return render(request, 'predict.html', {'prediction': prediction})  # No need for 'predictor/predict.html'
