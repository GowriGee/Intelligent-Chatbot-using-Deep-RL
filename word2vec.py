from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
dataframe=pd.read_csv('data.csv')
dataframe.head(3)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
app = Flask(__name__)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = ""
    
    for word in words:
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
        lemmas+=lemma + " "
    return lemmas

@app.route("/")
def index():
    return render_template("index1.html")

Q=dataframe['Q'].values.astype("str")
X=[]
for question in Q:
    words= nltk.word_tokenize(question)
    X.append(lemmatize_verbs(words))
#print (X)
Y=dataframe['A'].values.astype("str")
tokenizer = Tokenizer()
print
tokenizer.fit_on_texts( X ) 
tokenized_X = tokenizer.texts_to_sequences( X )
length_list = list()
for token_seq in tokenized_X:
    length_list.append( len( token_seq ))

max_input_length = np.array( length_list ).max()
print( 'Question max length is : ',max_input_length)

padded_X = pad_sequences( tokenized_X , maxlen=max_input_length , padding='post' )
encoder_input_data = np.array( padded_X )
print( 'Encoder input data shape : ',encoder_input_data.shape )

X_dict = tokenizer.word_index
print (X_dict)
num_X_tokens = len( X_dict )+1
print( 'Number of Input tokens : ', num_X_tokens)
new_Y=[]
for line in Y:
    new_Y.append( 'start ' + str(line) + ' end' )
new_Y[0]
tokenizer = Tokenizer(filters='!"#$&*+-;<=>[\\]^_`{|}~\t\n',split=' ',lower=False)
#tokenizer = Tokenizer()
tokenizer.fit_on_texts(new_Y) 
tokenized_Y = tokenizer.texts_to_sequences(new_Y)
length_list = list()
for token_seq in tokenized_Y:
    length_list.append( len( token_seq ))

max_output_length = np.array( length_list ).max()
print( 'Answer max length is : ',max_output_length)

padded_Y = pad_sequences( tokenized_Y , maxlen=max_output_length , padding='post' )
decoder_input_data = np.array( padded_Y )
print( 'Decoder input data shape : ',decoder_input_data.shape )

Y_dict = tokenizer.word_index
num_Y_tokens = len( Y_dict )+1
print( 'Number of Output tokens : ', num_Y_tokens)
decoder_target_data = list()
for token_seq in tokenized_Y:
    decoder_target_data.append( token_seq[ 1 : ] ) 
    
padded_Y = pad_sequences( decoder_target_data , maxlen=max_output_length, padding='post' )
onehot_Y = to_categorical( padded_Y , num_Y_tokens )
decoder_target_data = np.array( onehot_Y )
print( 'Decoder target data shape : ', decoder_target_data.shape )
import tensorflow
from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU, Input
from tensorflow.keras.models import Sequential, Model
encoder_inputs = Input(shape=( None , ))
encoder_embedding = Embedding( num_X_tokens, 256 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = LSTM( 128 , return_state=True  )( encoder_embedding )
encoder_states = [ state_h , state_c ]
# Decoder input

decoder_inputs = Input(shape=( None ,  ))
decoder_embedding = Embedding( num_Y_tokens, 256 , mask_zero=True) (decoder_inputs)
decoder_lstm = LSTM( 128 , return_state=True , return_sequences=True)

decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = Dense( num_Y_tokens , activation='softmax') 
output = decoder_dense ( decoder_outputs )
model = Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.summary()
history=model.fit([encoder_input_data , decoder_input_data], decoder_target_data, epochs=500 )
plt.plot(history.history['loss'])
import re
def make_inference_models():
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=( 128 ,))
    decoder_state_input_c = Input(shape=( 128 ,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
    return encoder_model , decoder_model
def str_to_tokens( sentence : str ):
    words = sentence.split()
    tokens_list = list()
    tokens_list1 = list()
    for word in words:
        lemmatizer = WordNetLemmatizer()
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
        tokens_list.append( X_dict[lemma ] ) 
        #tokens_list1.append( X_dict[lemma ] ) 
    #print (tokens_list1)
    return pad_sequences( [tokens_list] , maxlen=max_input_length , padding='post')
enc_model , dec_model = make_inference_models()

print("Hi, my name is Marcus. I am MBCET's chatbot. I am quite young and I am still learning. I assure you that I will be better over time:)")
#while(1):
@app.route("/get")
def get_bot_response():
    
    inp_quest=userText = request.args.get('msg')
    if(inp_quest.lower()=='bye'):
      return ("Thank you for talking. Goodbye!")
    inp_quest=re.sub(r"[?,/.!@%$#]", " ", inp_quest)
    try:
      states_values = enc_model.predict( str_to_tokens( inp_quest.lower()) )
      print (states_values)
      empty_target_seq = np.zeros( ( 1 , 1 ) )
      empty_target_seq[0, 0] = Y_dict['start']
      stop_condition = False
      decoded_answer = ''

      while not stop_condition :
          dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
          sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
          sampled_word = None
          for word , index in Y_dict.items() :
              if sampled_word_index == index :
                  if (word!='end'):
                    decoded_answer += ' {}'.format( word )
                  sampled_word = word
          
          if sampled_word == 'end' or len(decoded_answer.split()) > max_output_length:
              stop_condition = True
              
          empty_target_seq = np.zeros( ( 1 , 1 ) )  
          empty_target_seq[ 0 , 0 ] = sampled_word_index
          states_values = [ h , c ] 
      return decoded_answer
      #print( decoded_answer )
    except:
      return ("Sorry, didn't get your question")
      
#print("Thank you for talking. Goodbye!")

if __name__ == "__main__":
    app.run()
