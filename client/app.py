from flask import Flask,jsonify,request,render_template
app = Flask(__name__)
from sklearn.feature_extraction.text import CountVectorizer
import pickle

@app.route('/checkSpam',methods = ['GET','POST'])
def checkSpam():
	test = []
	text = request.form.get('data')
	test.append(text)
	with open('../models/spam_classifier.pkl','rb') as f:
		loaded_clf = pickle.load(f)
	with open('../models/text_vectorizer.pkl','rb') as f1:
		text_vectorizer1 = pickle.load(f1)  
	test_vector = text_vectorizer1.transform(test)
	return render_template("response.html",decision = loaded_clf.predict(test_vector)[0])

@app.route('/',methods = ['GET'])
def index():
	return render_template("index.html")
