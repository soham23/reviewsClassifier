from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append("../../Data")

from preprocessData import trainingTexts,trainingLabels,testTexts,testLabelsmples:]


model = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',DecisionTreeClassifier())])

parameters = {"clf__max_depth":range(1,10,2),"clf__min_samples_split":range(1,100,20)[1:]} 

gsModel = GridSearchCV(model,parameters)
gsModel = gsModel.fit(trainingTexts, trainingLabels)


print gsModel.best_score_
print gsModel.best_params_

joblib.dump(gsModel, "bestModel.pkl")