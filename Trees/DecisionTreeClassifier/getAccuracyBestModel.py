from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append("../../Data")

from preprocessData import trainingTexts,trainingLabels,testTexts,testLabels

model = joblib.load("bestModel.pkl")

pred = model.predict(testTexts)


print accuracy_score(pred,testLabels)