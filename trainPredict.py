#Importing the necessary libraries.
import numpy 
import string 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#Getting the dataset and the data preprocessing steps
filename="data_vtc_giao_duc.txt"
text=open(filename,encoding="utf8").read()
text=text.lower()
bad_chars = ['#', '*', '@', '_','\n','\\','//','`','-', '/','–', '\x02', '(', ')', "'", '”','“' ,'\ufeff',
 '‘', '’', '…', '⁄', '东', '买', '今', '他', '们', '天', '的', '西','!', '+', ':', ';', '>', '?',
 '\t', '%', '[', ']','‘', '’', '…', '̀', '́', '̂', '̃', '̆', '̉', '̛', '̣', 'δ',
  '•', '→', '、', '。', 'い', 'く', 'さ', 'し', 'た', 'っ', 'て', 'に', 'は', 'ま', 'を', 'ん', '族', '日','º','ð',
   '昨', '水', '私', '美', '行', '見', '館', '魚', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for i in range(len(bad_chars)):
    text = text.replace(bad_chars[i]," ")
text="".join(v for v in text if v not in string.punctuation)
words=sorted(set(text.split()))

#Indexing
char_to_int=dict((c,i) for i,c in enumerate(words))
int_to_char=dict((i,c) for i,c in enumerate(words))
print(char_to_int)
print(int_to_char)

#Preparing the dataset
seq_length=100
dataX=[]
dataY=[]
n_w=len(text)
for i in range(0,n_w - seq_length,1):
	seq_in=text[i:i + seq_length]
	seq_out=text[i + seq_length]
	X=[char_to_int[char] for char in seq_in]
	dataX.append(X)
	dataY.append((char_to_int[seq_out]))
n_patterns=len(dataX)

#Re-modelling the prepared dataset for the LSTM network
X=numpy.reshape(dataX, (n_patterns, seq_length,1))
X=X/float(n_w)
y=np_utils.to_categorical(dataY)

#Building the network
model=Sequential()
model.add(LSTM(256),input_shape=(X.shape[1], X.shape[2]), return_sequences=True)
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="modelPredict.hdf5"
checkpoint=ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbacks_list=[checkpoint]
model.fit(X,y,epochs=20,batch_size=128,callbacks=callbacks_list)