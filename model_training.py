
#%%
#1. Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
#2. Data loading
URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
df = pd.read_csv(URL)

# %%
#3. Data inspection
print("Shape of the data = ", df.shape)
print("\nInfo about the Dataframe\n",df.info())
print("\nDescription of the Dataframe\n",df.describe().transpose())
print("\nExample data:\n",df.head(1))

# %%
#4. Data cleaning
df.drop_duplicates()
print(df.info())

# %%
#5. Split the data into features and labels
features = df['review'].values
labels = df['sentiment'].values

#%%
#Convert labels into numerical form (label encoding)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# %%
#6. Perform train-test split
X_train,X_test,y_train,y_test = train_test_split(features,labels_encoded,train_size=0.8,random_state=42)

# %%
#7. Perform tokenization
#Define some parameters for the following preprocessing steps
vocab_size = 10000
embedding_dim = 16
max_length = 128
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

#(A) Define the Tokenizer object
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size,split=' ',oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

# %%
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

# %%
#(B) Transform text into tokens
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

# %%
#8. Perform padding and truncating
X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train_tokens,maxlen=max_length,padding=padding_type,truncating=trunc_type)
X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test_tokens,maxlen=max_length,padding=padding_type,truncating=trunc_type)

# %%
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
print(dict(list(reverse_word_index.items())[0:10]))

# %%
def decode_article(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])

print(decode_article(X_train_padded[3]))
print("----------------------")
print(X_train[3])

# %%
"""
For embedding, we can include it as a layer inside our Keras model
"""

#10. Model development
model = keras.Sequential()
#(A) Start with the embedding layer, this in return will also serve as the input layer for our deep learning model
model.add(keras.layers.Embedding(vocab_size,embedding_dim))

"""
For our deep learning model, we are going to construct a bidirectional LSTM, followed by a dense layer as classification layer, then lastly a dense layer for output layer.
"""

#(B) Build the bidirectional LSTM layer
model.add(keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim)))

#(C) Dense layer for classification and output
model.add(keras.layers.Dense(embedding_dim,activation='relu'))
model.add(keras.layers.Dense(len(np.unique(labels)),activation='softmax'))

model.summary()

# %%
#11. Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# %%
#12. Model training
n_epochs = 20
early_stopping = keras.callbacks.EarlyStopping(patience=3)
history = model.fit(X_train_padded,y_train,validation_data=(X_test_padded,y_test),epochs=n_epochs,callbacks=[early_stopping])

# %%
#13. Plot the graphs for training result
#(A) Loss graph
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training Loss','Test loss'])
plt.show()

# %%
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training Accuracy','Test Accuracy'])
plt.show()

# %%
#14. Model deployment
string_1 = 'This movie was not very well directed. they almost totally disregarded the book.I guess they were trying 2 save time. the only upside 2 me was that the actor who played finny was cute. Some of the dialog between the main characters appeared a little gay which was not the case in the book. Major parts of the book were once again chopped out.You lost the over all effect it was not as haunting as the book and left me lacking severely. Also the strong language although it was brief was very unnecessary. Also i was surprised ( not pleasantly) by a new character that was no where in the book.One of my favorite characters (leper) was poorly interpreted and portrayed. He seemed more sinister in the movie than the real leper was in the book. Over all disappointing.'
string_2 = 'This movie was terrible. The plot was terrible and unbelievable. I cannot recommend this movie. Where did this movie come from? This movie was not funny and wasted the talent of some great actors and actresses including: Gary Sinise, Kathy Bates, Joey Lauren Adams, and Jennifer Tilly.'

#(A) Convert the texts into tokens
token_1 = tokenizer.texts_to_sequences(string_1)
token_2 = tokenizer.texts_to_sequences(string_2)

# %%
#(B) Remove empty entries in the token list
def remove_space(token):
    temp = []
    for i in token:
        if i!=[]:
            temp.append(i[0])
    return temp

token_1_processed = np.expand_dims(remove_space(token_1),axis=0)
token_2_processed = np.expand_dims(remove_space(token_2),axis=0)

# %%
#(C) Perform padding and truncating
token_1_padded = keras.preprocessing.sequence.pad_sequences(token_1_processed,maxlen=max_length,padding=padding_type,truncating=trunc_type)
token_2_padded = keras.preprocessing.sequence.pad_sequences(token_2_processed,maxlen=max_length,padding=padding_type,truncating=trunc_type)

# %%
#(D) Put the padded tokens into the model for prediction
y_pred_1 = np.argmax(model.predict(token_1_padded))
y_pred_2 = np.argmax(model.predict(token_2_padded))
predictions = np.array([y_pred_1,y_pred_2])

# %%
#Use the label encoder inverse transform to obtain the class
class_predictions = label_encoder.inverse_transform(predictions)

# %%
y_pred_test = np.argmax(model.predict(X_test_padded),axis=1)
label_vs_prediction = np.stack([y_test,y_pred_test]).transpose()

# %%
#15. Save important components so that we can deploy our NLP model elsewhere
#(A) Tokenizer
import pickle
import os
PATH = os.getcwd()
tokenizer_save_path = os.path.join(PATH,"tokenizer.pkl")
with open(tokenizer_save_path,'wb') as f:
    pickle.dump(tokenizer,f)

# %%
#Check if the tokenizer can be loaded
with open(tokenizer_save_path,'rb') as f:
    tokenizer_loaded = pickle.load(f)

# %%
#(B) Keras model
model_save_path = os.path.join(PATH,"nlp_model")
keras.models.save_model(model,model_save_path)

# %%
#Check if the model can be loaded
model_loaded = keras.models.load_model(model_save_path)

#%%
model_loaded.summary()

# %%
label_encoder_save_path = os.path.join(PATH,"label_encoder.pkl")
with open(label_encoder_save_path,'wb') as f:
    pickle.dump(label_encoder,f)
