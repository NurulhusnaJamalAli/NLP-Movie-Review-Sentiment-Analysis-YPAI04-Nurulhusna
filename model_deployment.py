
#%%
#1. Import packages
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os,pickle,re
#%%
#2. Load the tokenizer and the NLP model
PATH = os.getcwd()

#3. Create functions to load the tokenizer, encoder and NLP model
def load_pickle_file(filepath):
    with open(filepath,'rb') as f:
        pickle_object = pickle.load(f)
    return pickle_object

@st.cache_resource
def load_model(filepath):
    model_loaded = keras.models.load_model(filepath)
    return model_loaded

#4. Define the file path towards the tokenizer and the model
tokenizer_filepath = os.path.join(PATH,'tokenizer.pkl')
model_filepath = os.path.join(PATH,'nlp_model')

#5. Load the tokenizer and the model using the functions
tokenizer = load_pickle_file(tokenizer_filepath)
model = load_model(model_filepath)

#6. Creating the streamlit app
#(A) A text to display what the app is about
st.title("Review Classification for 2 Types of Sentiment in a Movie Review")
#(B) Create an input text widget for users to type in the news
with st.form('input_form'):
    text_input = st.text_area("Input your reviews here:")
    submitted = st.form_submit_button("Submit")

text_inputs = [text_input]

#(C) Process the input string
#This function will remove unwanted string
def remove_unwanted_string(text_inputs):
    for index, data in enumerate(text_inputs):
        text_inputs[index] = re.sub('<.*?>',' ',data)
        text_inputs[index] = re.sub('[^a-zA-z]',' ',data).lower().split()
    return text_inputs
# a. Remove unwanted string using the function
text_removed = remove_unwanted_string(text_inputs)
# b. Tokenize the string
text_token = tokenizer.texts_to_sequences(text_removed)
# c. Padding and truncating
text_padded = keras.preprocessing.sequence.pad_sequences(text_token,maxlen=(200),padding='post',truncating='post')
# (D) Use the model to perform prediction
y_pred = np.argmax(model.predict(text_padded),axis=1)
# (E) Display the final result
#Load the label encoder
label_encoder_path = os.path.join(PATH,"label_encoder.pkl")
label_encoder = load_pickle_file(label_encoder_path)
label_map = {i:classes for i,classes in enumerate(label_encoder.classes_)}
result = label_map[y_pred[0]]
# (F) Write the prediction result into streamlit
st.write("The type of review sentiment is: " + result)
