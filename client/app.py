from flask import Flask, jsonify, request, render_template

app = Flask(__name__)
from sklearn.feature_extraction.text import CountVectorizer
import pickle

all_spam = []


@app.route("/checkSpam", methods=["GET", "POST"])
def checkSpam():
    test = []
    text = request.form.get("data")
    test.append(text)
    with open("../models/spam_classifier.pkl", "rb") as f:
        loaded_clf = pickle.load(f)
    with open("../models/text_vectorizer.pkl", "rb") as f1:
        text_vectorizer1 = pickle.load(f1)
    test_vector = text_vectorizer1.transform(test)
    decision = loaded_clf.predict(test_vector)[0]
    new_spam = {"decision": decision, "text_value": text}
    all_spam.append(new_spam)
    print(all_spam)
    return render_template("index.html", all_spam=all_spam)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", all_spam=all_spam)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
