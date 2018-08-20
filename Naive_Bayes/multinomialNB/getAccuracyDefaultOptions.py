from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from random import shuffle
import sys
sys.path.append("../../Data")

from preprocessData import trainingTexts,trainingLabels,testTexts,testLabels

model = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])


model.fit(trainingTexts, trainingLabels)

predictedLabels = model.predict(testTexts)

print accuracy_score(testLabels,predictedLabels)