from flask import Flask, request, render_template, redirect, url_for
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Disease descriptions (example data)
disease_descriptions = {
    "Common Cold": "A viral infection of the upper respiratory tract.",
    "Flu": "A contagious respiratory illness caused by influenza viruses.",
    "Migraine": "A severe headache often accompanied by nausea and sensitivity to light.",
    # Add more descriptions as needed
}

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        symptoms = request.form["symptoms"]
        symptoms_tfidf = vectorizer.transform([symptoms])
        prediction = model.predict(symptoms_tfidf)[0]
        description = disease_descriptions.get(prediction, "No description available.")
        return render_template("result.html", disease=prediction, description=description)
    return render_template("index.html")

@app.route("/feedback", methods=["POST"])
def feedback():
    feedback_text = request.form["feedback"]
    # Save feedback to a file or database (optional)
    with open("feedback.txt", "a") as f:
        f.write(feedback_text + "\n")
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)