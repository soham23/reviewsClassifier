from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from random import shuffle


model = joblib.load("bestModel.pkl")

text = raw_input("Enter a sample review : ")

result = model.predict([text])[0]

print "Positive review." if "2" in result else "Negative review."