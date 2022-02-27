import nltk
import pickle
from nltk.corpus import names
from nltk.corpus import brown
from nltk.classify import NaiveBayesClassifier

training = open('training_feats.pkl', "rb")
training_feats = pickle.load(training)
training_feats= list(training_feats)

testing = open('testing_feats.pkl', "rb")
testing_feats = pickle.load(testing)
testing_feats = list(testing_feats)

classifier = NaiveBayesClassifier.train(training_feats)

#test data
for i in range(len(testing_feats)):
    #gets the suffix, already in last three letters form
    temp = testing_feats[i][0]
    print(testing_feats[i][0], classifier.classify(temp))

print(nltk.classify.accuracy(classifier, testing_feats))
classifier.show_most_informative_features(10)