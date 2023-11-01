 # Load libraries
import pandas as pd
import string
import re
import pickle
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from afinn import Afinn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to clean and preprocess tweet text
def text_preprocessing(reviews):
    # Convert to lowercase
    reviews = reviews.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    reviews_without_punctuations = reviews.translate(translator)
    reviews_without_punctuations = re.sub(r'\d+', '', reviews_without_punctuations)
    
    # Tokenization
    tokens = word_tokenize(reviews_without_punctuations)
 
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]
    
    cleaned_reviews = ' '.join(lemmatized_tokens).strip()
    
    return cleaned_reviews

def main():
    st.title('Welcome to Sentiment Analysis!')
    
    input_ = st.text_area('Enter the text')
    
    tfidf = pickle.load(open('C:\\Users\\Dell\\OneDrive\\Desktop\\P297 Project\\final_deploy\\tfidf.pkl','rb'))

    # Load the final model
    
    model = pickle.load(open('C:\\Users\\Dell\\OneDrive\\Desktop\\P297 Project\\final_deploy\\Final_model.pkl', 'rb'))

    if st.button('Submit'):
        # Input from user
        cleaned_input = text_preprocessing(input_)
        # Converting text to tfidf vectorizer
        array_input = tfidf.transform(pd.Series(cleaned_input)).toarray()
        # Prediction
        result = model.predict(array_input)
        # Result
        positive_emoji = "\U0001F604"  # Positive sentiment
        negative_emoji = "\U0001F61E"  # Negative sentiment
        neutral_emoji = "\U0001F610"   # Neutral sentiment
        if result == 0:
            st.header(f'Sentiment : Negative {negative_emoji}')
        elif result == 1:
            st.header(f'Sentiment : Neutral {neutral_emoji}')
        elif result == 2:
            st.header(f'Sentiment : Positive {positive_emoji}')
        
        afinn = Afinn()
        word_list = cleaned_input.split(" ")
        pos_word = []
        neg_word = []
        neutral_word = []
        for i in word_list:
            score = afinn.score(i)
            if score > 0:
                pos_word.append(i)
            elif score == 0:
                neutral_word.append(i)
            elif score < 0:
                neg_word.append(i)
        
        st.subheader('Positive Keywords:')
        st.write(pos_word)  # Display positive words
        st.subheader('Neutral Keywords:')
        st.write(neutral_word)  # Display neutral words
        st.subheader('Negative Keywords:')
        st.write(neg_word)  # Display negative words

if __name__ == "__main__":
    main()
