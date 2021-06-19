# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "data_vtc_giao_duc.txt"
#filename = "full.txt"

raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
print("cac")
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))



filename = 'weights-improvement-74-1.4272.hdf5'

model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


#predict
original_text = ''
predicted_text=[]
def cls():
    print("\n" * 50)
while True:
   # print(original_text)
    original_text = original_text.replace("  ", " ")  # fix 2 dau cach

    text = input(original_text+" ")
    #
    #
    #
    #cls()
    print("\n\n") 
    text = text.lower()  ## khoi vang loi uppercase

    if text=='`':     #      ~`  > use predict nhu new input
        text= predicted_text[len(predicted_text)-1].replace(" ","")
    if text=='//':
        break
    inp= list(original_text+' '+text)
    inp.pop(0)
#    print('------------\n predicted_text: ',predicted_text)
#    print('original_text: ', original_text)
 #   print('inp: ', inp) ##

    last_word = inp[len(original_text):]
    inp = inp[:len(original_text)]    
    original_text = original_text+' '+text
    last_word.append(' ')
#    print('inp ',inp)
#    print('last_word ',last_word)
#    print('original_text: ', original_text)
    ########
    inp_text = [char_to_int[c] for c in inp]
   
    last_word = [char_to_int[c] for c in last_word]
   
 #   print('inp_text ',inp_text)
#    print('last_word ', last_word)

    if (len(inp_text) > 100):
        inp_text = inp_text[len(inp_text)-100: ]
    if len(inp_text) < 100:
        pad = []
        space = char_to_int[' ']
        pad = [space for i in range(100-len(inp_text))]
        inp_text = pad + inp_text
    
    while len(last_word)>0:
        X = np.reshape(inp_text, (1, SEQ_LENGTH, 1))
        next_char = model.predict(X/float(VOCABULARY))
        inp_text.append(last_word[0])
        inp_text = inp_text[1:]
        last_word.pop(0)
       # print(int_to_char[np.argmax(next_char)])
 #   print('inp_text ',inp_text)
  #  print('last_word ', last_word)
    next_word = []
    next_char = ':'
    while next_char != ' ':
        X = np.reshape(inp_text, (1, SEQ_LENGTH, 1))
        next_char = model.predict(X/float(VOCABULARY))
        index = np.argmax(next_char)        
        next_word.append(int_to_char[index])
        inp_text.append(index)
        inp_text = inp_text[1:]
        next_char = int_to_char[index]
    
    predicted_text = predicted_text + [''.join(next_word)]
    print("(Du doan: " + ''.join(next_word), end=')')
    

from tabulate import tabulate

original_text = original_text.split()
predicted_text.insert(0,"")
predicted_text.pop()

table = []
dem=0
for i in range(len(original_text)):
    if (original_text[i].replace(" ","")==predicted_text[i].replace(" ","")):
        dem=dem+1
        table.append([original_text[i], predicted_text[i], str(dem)])
    else:
        table.append([original_text[i], predicted_text[i], 'flase'])
print(tabulate(table, headers = ['Actual Word', 'Predicted Word', 'Resutf']))
