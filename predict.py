from joblib import load

# Load model
model = load('model.joblib')

# TEST
if model:
    print("✅ Model has been initialised successfully!")