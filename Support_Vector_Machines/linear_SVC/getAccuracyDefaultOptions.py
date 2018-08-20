from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
import sys
sys.path.append("../../Data")

from preprocessData import trainingTexts,trainingLabels,testTexts,testLabels

model = Pipeline([("vectorize",CountVectorizer()),('ifdif',TfidfTransformer()),('classification', LinearSVC())])

model.fit(trainingTexts, trainingLabels)

predictedLabels = model.predict(testTexts)

print accuracy_score(predictedLabels,testLabels)