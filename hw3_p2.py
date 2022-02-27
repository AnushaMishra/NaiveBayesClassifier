import nltk
import random
from nltk.corpus import names
from nltk.corpus import brown
from collections import defaultdict
import pickle

all = brown.tagged_words(tagset='universal')
training_set = []
testing_set = []
rand_num = []

def create_randlist(rand_list, all_list):
    while(len(rand_num)<1000) :
        rand_index = random.randint(0, len(all_list)-1)
        rand_list.append(rand_index)
        list(set(rand_list))
    return rand_list

def create_set(rand_list, all_list, n):
    temp_set = []
    while(len(temp_set)<n):
        temp_set.append(all[rand_list.pop()])
    return temp_set

#def last_three(word, tag):
    #return {word[-3:]: tag}
def gender_features(word):
    return {'last_letter': word[-1]}
    
def last_three(word):
    return {'last_three': word[-3:]}

def feat_set(tuple_list):
    featuresets = [(last_three(word), tag) for (word, tag) in tuple_list]
    return featuresets

#Added this function
def transform_tup(tuples):
    output = [(x[-3:], y) for (x,y) in tuples]
    return output

#Updated this function
def same(training_set):
    output = []
    #beginning of code from: https://www.geeksforgeeks.org/python-group-tuples-in-list-with-same-first-value/
    mapp = defaultdict(list)
    for x, y in training_set:
        mapp[x].append(y)
    r = [(x, *y) for x, y in mapp.items()]
    #end of code from: https://www.geeksforgeeks.org/python-group-tuples-in-list-with-same-first-value/
    #print(r)
    for x in range(len(r)):
        #mapped to more than one tag
        if(len(r[x])>2):
            #all of the multiple tags are not the same
            if(len(tuple(set(r[x]))) > 2):
                temp = tuple([r[x][0], 'X'])
                #print(temp)
                output.append(temp)
                #r[x] = temp
            #all of the multiple tags are the same, so just map to first one
            else:
                temp = tuple([r[x][0], r[x][1]])
                #print(temp)
                output.append(temp)
                #r[x] = temp
        else:
            output.append(r[x])
    return output

def feature_list(tuple_list):
    features = same(tuple_list)
    features = feat_set(tuple_list)
    return features
#({'*suffix*': tag}, tag)
rand_num = create_randlist(rand_num, all)
training_set = create_set(rand_num, all, 900)
testing_set = create_set(rand_num, all, 100)

#added following two lines
training_set = transform_tup(training_set)
testing_set = transform_tup(testing_set)
features = feature_list(training_set)
features_testing = feature_list(testing_set)

with open('training_feats.pkl', 'wb') as training:
    pickle.dump(features, training)
training.close()
with open('testing_feats.pkl', 'wb') as testing:
    pickle.dump(features_testing, testing)
testing.close()


print("Training:")
print(features)
print("Testing:")
print(features_testing)