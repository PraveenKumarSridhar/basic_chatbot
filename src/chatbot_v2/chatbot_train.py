import warnings,logging,os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
except:
    pass
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import random,json,os,pickle
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

# from config.global_vars import *

# * Read intents file
with open('config/intents.json') as f:
    intents = json.load(f)

print(intents)
# * Create vocabulary
labels,docs_x,docs_y = [],[],[]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        docs_x.append(pattern)
        docs_y.append(intent['tag'])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

n_class = len(labels)
docs_x = np.array(docs_x)
doc_labels = np.array([labels.index(y) for y in docs_y])
print(doc_labels)

# * Tokenize and pad the seq,
VOCAB_SIZE=1000
maxlen =20
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(docs_x)
sequences = tokenizer.texts_to_sequences(docs_x)
data = pad_sequences(sequences, maxlen=maxlen)
TOKENIZERS = 'tokenizer/'

if not os.path.exists(TOKENIZERS):
    os.makedirs(TOKENIZERS)

#* Save the tokenizer.
tokenizer_path = os.path.join(TOKENIZERS, 'tokenizer.pickle')
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
max_features = len(word_index)+1

#* convert the labels as a categorical var
target = to_categorical(doc_labels,num_classes=n_class)


#* Define model
deep_inputs = Input(shape=(maxlen,))
embedding = layers.Embedding(max_features, 200, input_length = 20 ,trainable=True)(deep_inputs)
spd1 = layers.SpatialDropout1D(0.3)(embedding)
bi_lstm = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(spd1)
dense1 = layers.Dense(16)(bi_lstm)
out_layer = layers.Dense(n_class, activation='softmax')(dense1)
model = Model(inputs=deep_inputs, outputs=out_layer)
        

model.summary()

# * Compilethe model
model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics = ['acc'])

# * Train and save the model.
model_path = '/home/praveenkumar/Documents/CODE/github_repos/basic_chatbot/basic_chatbot/src/model/v2/bilstm.h5'
model_chkpt = ModelCheckpoint(model_path,monitor = 'acc',mode = 'auto',verbose=1)
history  = model.fit(data,target,batch_size=4,epochs=100,callbacks=[model_chkpt])

