from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append("../../Data")

from preprocessData import trainingTexts,trainingLabels,testTexts,testLabels


model = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',SGDClassifier())])

parameters = {"clf__alpha":[0.0001],"clf__n_iter":[10,100,1000,10000]} 

gsModel = GridSearchCV(model,parameters)
gsModel = gsModel.fit(trainingTexts, trainingLabels)


print gsModel.best_score_
print gsModel.best_params_

joblib.dump(gsModel, "bestModel.pkl")