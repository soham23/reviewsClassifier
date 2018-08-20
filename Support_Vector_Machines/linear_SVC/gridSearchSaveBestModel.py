from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import sys
sys.path.append("../../Data")

from preprocessData import trainingTexts,trainingLabels,testTexts,testLabels


model = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', LinearSVC())])

parameters = {'clf__C':(0.01,0.03,0.1,0.3,1,3,10)} 

gsModel = GridSearchCV(model,parameters)
gsModel = gsModel.fit(trainingTexts, trainingLabels)


print gsModel.best_score_
print gsModel.best_params_

joblib.dump(gsModel, "bestModel.pkl")