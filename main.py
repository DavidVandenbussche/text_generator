import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Activation, Dense, LSTM

# text = open('Shakespear.txt').read().lower() # reading the file and lowering the letters
text = open('Proust.txt', encoding = 'utf-8').read().lower()

text = text[10:508400] # training data

characters = sorted(set(text)) # set select all the different characters, and sorted will sort them to associate them to numbers

# now creating 2 dict to convert char to index and index to char 
char_to_index = dict((c, i) for i, c in enumerate(characters)) # char are the keys so we find an index when giving a char
index_to_char = dict((i, c) for i, c in enumerate(characters)) # finding a char when given an index

SEQ_LENGTH = 40 # how much character should be used to predict the next one
STEP_SIZE = 3 # numbers of characters we are moving to create the next sequence

# comment/uncomment to re load the model
'''
sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE): # moving in the text based on what we decided
    sentences.append(text[i: i + SEQ_LENGTH]) # filling up the sentences, the last one is not included
    next_char.append(text[i + SEQ_LENGTH]) # the last one is the next char that we want to guess 

# now we need two arrays of numerical values
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_) # 3D array, all possible sentences, 
                                                                            # all possible position in the sequence, 
                                                                            # and all possible characters, when one occurs
                                                                            # it becomes 1, this is the training data
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_) # target data, predicting in sentence number X that the next char is Y



for i, sentence in enumerate(sentences): # browsing sentences
    y[i, char_to_index[next_char[i]]] = 1 # for sentence i the next char is the one with a 1
    for p, char in enumerate(sentence): # browsing characters
        x[i, p, char_to_index[char]] = 1 # for sentence i position t and character number X the block becomes 1
    

model = Sequential()
model.add(LSTM(128, input_shape = (SEQ_LENGTH, len(characters)))) # we get the input and immediately put them in th LSTM layer
model.add(Dense(len(characters)))
model.add(Activation('softmax')) # softmax = sum up to 1 ie probabilities

model.compile(loss='categorical_crossentropy',optimizer = RMSprop(lr=0.01))

model.fit(x, y, batch_size = 256, epochs = 10)
model.save('textgen.model')
'''

model = tf.keras.models.load_model('textgen.model')

# function from the official keras tutorial website, use prediction to create sentences,
# the higher the temperature the riskier it becomes (choosing not only the max arg in softmax)
# hence creating more creative sentences, but if too high doesn't makes sense

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1) # create an input from Shakespear text at random
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH] # the sentece that we start with
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters))) # once again numpy array with 0, 1 sentence this time
        for p, character in enumerate(sentence):
            x[0, p, char_to_index[character]] = 1 # filling up the input data with position and character

        predictions = model.predict(x, verbose =0)[0] # verbose is which way you want to see the animation (0,1,2)
                                                      # we get the softmax number of the prediction
        next_index = sample(predictions, temperature) # this gives us the next index of the character
        next_character = index_to_char[next_index]

        generated += next_character # now we will add it to the input for the next character
        sentence = sentence[1:] + next_character # shifting the sentence from 1 character to predict the next one
    return generated

print('--------0.6----------')
print(generate_text(500, 0.6))

 