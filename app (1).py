from flask import Flask, request, render_template_string
import pickle

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IMDb Sentiment Analysis</title>
</head>
<body>
    <h2>IMDb Review Sentiment Prediction</h2>
    <form method="post">
        <textarea name="review" rows="8" cols="80" placeholder="Enter movie review"></textarea><br><br>
        <input type="submit" value="Predict">
    </form>

    {% if prediction %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        review = request.form["review"]

        # IMPORTANT:
        # This works only if your model was trained directly on numeric features.
        # If you used TF-IDF/CountVectorizer, load the vectorizer.pkl and transform
        # the review before prediction.
        try:
            prediction = model.predict([review])[0]
        except Exception as e:
            prediction = f"Model requires preprocessing/vectorizer: {e}"

    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
