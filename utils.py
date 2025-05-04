import re
import spacy 
import nltk
from nltk.corpus import stopwords
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import joblib

################################ PRE-PROCESSING ################################
# Descargamos los stopwords de nltk
nltk.download('stopwords')

# Cargamos el modelo de Spacy
lemmatizer = spacy.load('en_core_web_sm')

    # Guardo las stopwords en un set
stop_words = set(stopwords.words('english'))

def clean_text(text):
    
    """
    Recibe un texto y lo procesa (Minúsculas, eliminación de caracteres innecesarios y doles espacios, eliminación de stopwords y lematización.)
    Args:
        text(string): Texto crudo
    Returns:
        text(string): Devuelve el texto procesado
    """
    # Pasamos los caracteres a minúsculas
    text = text.lower()
    
    # Eliminamos etiquetas HTML 
    text = re.sub(r'<.*?>', ' ', text)
    
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

################################ Funciones de archivo ################################
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


################################ Funciones de métricas ################################

def plot_confussion_matrix(y_true, y_pred, labels=None):
    """
    Función para graficar la matriz de confusión 
    Args:
        y_true (numpy.ndarray): Array que contiene los valores reales
        y_pred (numpy.ndarray): Array que contiene los valores predichos
        labels (list): Etiquetas de las clases
    """
    # Usamos el metodo `confusion_matrix` de sklearn.metrics
    matrix = confusion_matrix(y_true, y_pred)
    
    # Ploteamos la matriz para que se vea mejor
    plt.figure(figsize=(8, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()


def topn_most_similar_plot(word, model, topn=10, ndim=2):
    """
        Función para graficar las top-n palabras más parecidas a una palabra en 2 o 3 dimensiones
    Args:
        word (string): Palabra que se usará como referencia 
        model (gensim.models.Word2Vec): Modelo Word2Vec entrenado
        topn (int, optional): Número máximo de palabras similares a la palabra de referencia a tomar en cuenta. Por defecto 10.
        ndim (int, optional): Número de dimensiones en las que se graficará. Por defecto 2.
    """
    # En caso de que el usuario introduzca una dimensión no validad devolvimos un mensaje
    if ndim not in range(2,4):
        return print('Ingresa una dimensión validad (2 o 3)')
    
    # Verificamos que la palabra se encuentre en nuestro vocabulario, sino devolvemos una advertencia
    if word.strip().lower() not in model.wv.key_to_index:
        print(f'La palabra "{word}" no se encuentra en el vocabulario.')
        return
        
    # Buscamos las top-n palabras más similares
    similar_words = model.wv.most_similar(word.strip().lower(), topn=min(topn, len(model.wv) - 1))
        
    # Creamos un array en el que almacenaremos el embedding de cada palabra
    similar_vecs = np.array([model.wv[word] for word, _ in similar_words])
    
    # A esos embeddings agregamos el embedding de la palabra de referencia
    similar_vecs = np.vstack([model.wv[word], similar_vecs])
    
    # Aplicamos una técnica de reducción de dimensionalidad en este caso TNSE 
    tnse = TSNE(n_components=ndim, perplexity=5, random_state=42).fit_transform(similar_vecs)
    
    # Tomamos las palabras
    words = [word for word, _ in similar_words] + [word]
    
    # Graficamos
    fig = plt.figure(figsize=(7, 5))
    
    plt.title(f'Top {topn} palabras más similares a "{word}" ({ndim}D)')
    plt.grid(True)
    projection = "3d" if ndim == 3 else None
    ax = fig.add_subplot(111, projection=projection)
    for vec, word in zip(tnse, words):
        if ndim == 2:
            x, y = vec
            ax.scatter(x, y)
            ax.text(x - 5, y + 2, word)
        else:
            x, y, z = vec
            ax.scatter(x, y, z)
            ax.text(x - 5, y + 2, z + 3, word)
    plt.tight_layout()
    plt.show()


################################ Funciones para modelos ################################
# Ruta para el almacenamiento de modelos 
MODELS_FOLDER_PATH = './Models'

# Ruta de modelos usados en ML
ML_MODELS_FOLDER_PATH = path_concat(MODELS_FOLDER_PATH, 'MachineLearning')

# Ruta de modelos usados en DL
DL_MODELS_FOLDER_PATH = path_concat(MODELS_FOLDER_PATH, 'DeepLearning')

# Función para verificar si el tipo de modelo es valido
def is_valid_type(model_type):
    # Estandarizamos el string
    model_type = model_type.lower()
    
    # En caso de que no sea valido lanzamos un error
    if model_type not in ['ml', 'dl']:
        raise ValueError('Ingresa un tipo de modelo valido! ("ml" o "dl")')
    
    # Si todo ha salido bien devolvemos el tipo del modelo
    return model_type

# Función para obtener la ruta de un modelo
def get_model_path(model_name, model_type):
    # Verificamos el tipo de modelo
    model_type = is_valid_type(model_type)
    
    # De acuerdo al tipo asignamos la ruta
    base_path = ML_MODELS_FOLDER_PATH if model_type == 'ml' else DL_MODELS_FOLDER_PATH
    
    # Concatenamos el nombre con la ruta
    model_path = path_concat(base_path, model_name)
    
    # Devolvemos la ruta del modelo
    return model_path

# Función que verifica que las carpetas donde se almacenaran los modelos existe, sino las crea
def does_models_path_exists():
    # Iteramos sobre la lista con las carpetas necesarias
    for path in [MODELS_FOLDER_PATH, ML_MODELS_FOLDER_PATH, DL_MODELS_FOLDER_PATH]:
        
        # En caso de que la ruta no exista la creamos
        if not path_exists(path):
            make_dir(path, True)

# Función para guardar modelos
def save_model(model, model_name, model_type):
    """
        Esta función guarda los modelos en rutas predeterminadas
    Args:
        model (model): Objeto modelo
        model_name (string): Nombre del modelo
        model_type (string): Tipo de modelo (ml | dl)
    """
    # Antes de guardar un modelo verificamos que exista la ruta 
    does_models_path_exists()
    
    #  Obtenemos la ruta del modelo
    model_path = get_model_path(model_name, model_type)
    
    # De acuerdo al tipo de modelo lo guardamos de una u otra forma
    if model_type == 'ml':
        joblib.dump(model, model_path)
    else:
        pass
    
    return('Modelo guardado correctamente!')

# Función para cargar un modelo
def load_model(model_name, model_type):
    """
        Esta función carga los modelos guardados por la función `save_model`

    Args:
        model_name (string): Nombre del modelo
        model_type (string): Tipo del modelo

    Raises:
        FileNotFoundError: Error en caso de que el modelo no exista

    Returns:
        model: Devuelve el objeto modelo
    """
    #  Obtenemos la ruta del modelo
    model_path = get_model_path(model_name, model_type)
    
    # Verificamos que el modelo exista, sino lanzamos un error
    if not path_exists(model_path):
        raise FileNotFoundError('El modelo no existe!')

    # De acuerdo al tipo de modelo lo cargamos de una u otra forma
    if model_type == 'ml':
        model = joblib.load(model_path)
    else:
        pass
    return model

# Función que verifica si un modelo existe
def model_exists(model_name, model_type):
    """
        Esta función devuelve un `Bool` que indica si el modelo existe o no

    Args:
        model_name (string): Nombre del modelo
        model_type (string): Tipo del modelo

    Returns:
        bool: Indica si existe o no el modelo
    """
    # Obtenemos  la ruta del modelo
    path = get_model_path(model_name, model_type)
    
    # Verificamos si la ruta existe
    isExisting = path_exists(path)
    
    # Devolvemos un booleano que indique su existencia
    return isExisting


# Función para testear los modelos con diccionarios reviews propias
def test_model_dict(samples_by_label, model, codifier, label_names):
    """
        Recibe un diccionario en el cual iterara sobre cada llave para predecir la clase de cada una de las reviews que contiene
    Args:
        samples_by_label (dict): Diccionario (Label: [Reviews])
        model (model): _description_
        codifier (sklearn.feature_extraction.text.TfidfVectorizer): Codificador
        label_names (list): Lista las categorías en orden 
    """
    print(f'{"Valor real":<20}{"Valor predicho":>20}')
    # Iteramos sobre el diccionario
    for real_label, reviews in samples_by_label.items():
        # Ahora iteramos en la lista de reviews
        for review in reviews:
            # Entadarizamos la review
            cleaned = clean_text(review)
            
            # Codificamos la review
            vectorized = codifier.transform([cleaned])
            
            # Usamos el modelo para predecir en función de la review codificada
            pred = model.predict(vectorized)[0]
            
            # Mostramos una comparación entre la clase real y la predicha
            print(f'{real_label:<20}{label_names[pred]:>20}')

# Función para testear los modelos con UNA review propia

def test_model(review, model, codifier, label_names):
    """
        Recibe un diccionario en el cual iterara sobre cada llave para predecir la clase de cada una de las reviews que contiene
    Args:
        samples_by_label (dict): Diccionario (Label: [Reviews])
        model (model): _description_
        codifier (sklearn.feature_extraction.text.TfidfVectorizer): Codificador
        label_names (list): Lista las categorías en orden 
    """

    cleaned = clean_text(review)
            
    # Codificamos la review
    vectorized = codifier.transform([cleaned])
            
    # Usamos el modelo para predecir en función de la review codificada
    pred = model.predict(vectorized)[0]
            
    # Mostramos una comparación entre la clase real y la predicha
    return label_names[pred]

