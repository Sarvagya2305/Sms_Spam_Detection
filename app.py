import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def transform_text(text):
    text=text.lower() # change into lower letter 
    text=nltk.word_tokenize(text)# tokenize each words
    y=[]
    for i in text: # removing spl character 
        if i.isalnum(): 
            y.append(i)
    text=y[: ]
    y.clear()
    for i in text:# removing stopwords and puncutation marks
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:  # for stemming the words
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))  # rb = read binary

st.title("Email/SMS spam Classifier")

input_sms = st.text_area("Enter the message")


if st.button('predict'):
    # 1. preprocess
    transform_sms = transform_text(input_sms)  # Corrected the function name
    # 2. vectorize
    vector_input = tfidf.transform([transform_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



