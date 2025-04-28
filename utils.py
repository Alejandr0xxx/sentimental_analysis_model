import re
import spacy 
import nltk
from nltk.corpus import stopwords
import os

# Descargamos los stopwords de nltk
nltk.download('stopwords')

# Cargamos el modelo de Spacy
lemmatizer = spacy.load('en_core_web_sm')

    # Guardo las stopwords en un set
stop_words = set(stopwords.words('english'))

def limpiar_texto(text):
    
    """
    Recibe un texto y lo procesa (Minúsculas, eliminación de caracteres innecesarios y doles espacios, eliminación de stopwords y lematización.)
    Args:
        text(string): Texto crudo
    Returns:
        text(string): Devuelve el texto procesado
    """
    
    # Pasamos los caracteres a minúsculas
    text = text.lower()
    
    # Eliminamos caracteres no alfabéticos
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Eliminamos multiples espacios
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Eliminamos las stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Usamos spicy para lematizar el texto
    lemma = lemmatizer(text)
    
    # Juntamos los lemas de las palabras en un solo texto
    text = ' '.join([token.lemma_ for token in lemma])
    
    return text

# Para evitar redundancia al importar os en todas los notebooks mejor creo una función aquí y será aquí el único lugar en el que la importe
#Función para saber si una ruta ya existe
def path_exists(path):
    isExisting = os.path.exists(path)
    
    return isExisting

#Función para unir dos rutas
def path_concat(pathOne, pathTwo):
    final_path = os.path.join(pathOne, pathTwo)
    return final_path

# Función para crear un directorio
def make_dir(path, exist_ok):
    os.makedirs(path, exist_ok=exist_ok)