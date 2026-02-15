import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("model/iris_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Create input as DataFrame with same column names used during training
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
)

# Scale input
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

print("Predicted Iris Species:", prediction[0])
