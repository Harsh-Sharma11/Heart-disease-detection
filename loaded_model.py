import joblib

filename = 'heart_disease_detection.sav'
loaded_model = joblib.load(filename)

y_pred = loaded_model.predict([[1, 50, 1, 1, 30, 1, 1, 1, 1, 300, 220, 180, 35, 40, 100]])
print(y_pred)