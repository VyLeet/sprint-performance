from flask import Flask, render_template, request
import joblib
from sklearn.linear_model import LinearRegression

model = joblib.load("model.joblib")

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the form
    text1 = float(request.form["text1"])
    text2 = float(request.form["text2"])
    text3 = float(request.form["text3"])
    text4 = float(request.form["text4"])
    text5 = float(request.form["text5"])
    text6 = float(request.form["text6"])
    text7 = float(request.form["text7"])
    text8 = float(request.form["text8"])

    # Make prediction using the loaded model
    prediction = model.predict(
        [[text1, text2, text3, text4, text5, text6, text7, text8]]
    )

    # Return the prediction result to the user
    return render_template("result.html", prediction=prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
