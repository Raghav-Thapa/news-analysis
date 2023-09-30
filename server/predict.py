import joblib
import sys
import json

# Load the model (replace 'model.pkl' with your model file path)
model = joblib.load('./jupyter.pkl')

def predict_sentiment(text):
    # Perform preprocessing and feature extraction if needed
    # Make predictions using the loaded model
    prediction = model.predict([text])[0]
    return prediction

if __name__ == '__main__':
    input_data = json.loads(sys.argv[1])
    result = predict_sentiment(input_data)
    print(json.dumps(result))
