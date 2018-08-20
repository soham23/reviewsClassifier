from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append("../../Data")

from preprocessData import trainingTexts,trainingLabels,testTexts,testLabels


model = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])

parameters = {'clf__alpha':(0.01,0.03,0.1,0.3,1,3,10)} 

gsModel = GridSearchCV(model,parameters)
gsModel = gsModel.fit(trainingTexts, trainingLabels)


print gsModel.best_score_
print gsModel.best_params_

joblib.dump(gsModel, "bestModel.pkl")