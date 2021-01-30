import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.stem.lancaster import LancasterStemmer
import random,json,tflearn,os,pickle
import tensorflow as tf
import numpy as np
# nltk.download('punkt')
stemmer = LancasterStemmer()

# * Read intents file
with open('config/intents.json') as f:
    intents = json.load(f)

# * Create vocabulary
vocab,labels,docs_x,docs_y = [],[],[],[]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        vocab.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

vocab = sorted(list(set([stemmer.stem(word.lower()) for word in vocab if word not in ['?','.',','] ])))
labels = sorted(labels)

print('Uniques words in vocab:',vocab)

# * Create input and output for training using one hot encoding
training,output = [],[]
out_empty = [0 for _ in range(len(labels))]

for index,doc in enumerate(docs_x):
    
    wrds = [stemmer.stem(w) for w in doc]
    bow = [1 if w in wrds else 0 for w in vocab]

    output_row = out_empty[:]
    output_row[labels.index(docs_y[index])] =1

    training.append(bow)
    output.append(output_row)

training,output = np.array(training),np.array(output)

#  * Saving the data req for reconstruction the model 
with open('model/v1/data.pickle','wb') as f:
    pickle.dump((vocab,labels,training,output),f)
    
# ! Deprecated tf method
# tf.reset_default_graph()
# * Initializing the model
net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# * Fitting the model
model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
os.system('play -nq -t alsa synth {} sine {}'.format(0.5, 440))

# * Save the model
model.save('model/v1/chabot_v1.tflearn')
