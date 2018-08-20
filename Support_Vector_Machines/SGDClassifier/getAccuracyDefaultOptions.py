from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
import sys
sys.path.append("../../Data")

from preprocessData import trainingTexts,trainingLabels,testTexts,testLabels

trainingLabels = map(lambda x: 1 if "1" in x else 0,trainingLabels)
testLabels = map(lambda x: 1 if "1" in x else 0,testLabels)


model = Pipeline([("vectorize",CountVectorizer()),('ifdif',TfidfTransformer()),('classification', SGDClassifier())])


model.fit(trainingTexts, trainingLabels)

predictedLabels = model.predict(testTexts)

print accuracy_score(predictedLabels,testLabels)