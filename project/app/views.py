from django.shortcuts import render

# Create your views here.


from django.shortcuts import render
from joblib import load
import numpy as np
import os
from django.shortcuts import render
from joblib import load
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), 'weather_model.joblib')
model = load(model_path)

def weather_predictor(request):
    prediction = None
    error = None
    
    if request.method == 'POST':
        try:
            # Get and validate form data
            cloud = int(request.POST.get('cloud', 1))
            humidity = int(request.POST.get('humidity', 50))
            temp = int(request.POST.get('temperature', 20))
            
            if not 0 <= humidity <= 100:
                raise ValueError("Humidity must be between 0-100%")
            if not -50 <= temp <= 50:
                raise ValueError("Temperature must be between -50°C and 50°C")
            
            # Make prediction
            input_data = np.array([[cloud, humidity, temp]])
            prediction = int(model.predict(input_data)[0])
            
        except Exception as e:
            error = str(e)
    
    return render(request, 'index.html', {
        'prediction': prediction,
        'error': error,

    })