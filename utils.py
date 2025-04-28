import re
import spacy 
from nltk.corpus import stopwords

lemmatizer = spacy.load('en_core_web_sm')

stopwords = set(stopwords.words('english'))

def limpiar_texto(text):
    text = text.lower()
    
    text = re.sub(r'[^a-s\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = ' '.join([word for word in text.split() if word not in stopwords])
    
    lemma = lemmatizer(text)
    
    text = ' '.join([word.lemma_ for word in lemma])
    
    return text