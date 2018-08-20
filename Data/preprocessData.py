from random import shuffle

fil = open("E:\\Documents\\MYCODE\\Machine_Learning\\CompleteProjects\\SingleTagClassifier3\\Data\\amazonLabels.txt")
dt = fil.readlines()
fil.close()


randomize = False

if(randomize):
    shuffle(dt)

allTexts = [i[i.index(" ")+1:-1] for i in dt]
allLabels = [i[:i.index(" ")] for i in dt]

allLabels = map(lambda x: 1 if "1" in x else 0,allLabels)	# label_1 becomes 0, i.e. negative review -> 0 and positive review -> 1


noOfExamples = len(allTexts)
noOfTrainingExamples = int(noOfExamples * 0.8)


trainingTexts, trainingLabels = allTexts[:noOfTrainingExamples], allLabels[:noOfTrainingExamples]
testTexts, testLabels = allTexts[noOfTrainingExamples:], allLabels[noOfTrainingExamples:]