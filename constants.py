from utils import path_concat
#Debido al gran tamaño de los datos tomaré solo tomaré las reviews de "Videogames", "Electronica" y "Teléfono y accesorios" para luego realizar un muestreo 
DATASETS_URL = ['https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz', 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz', 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Cell_Phones_and_Accessories.jsonl.gz']

#Ruta de destino donde se guardaran los archivos con las reviews
DATASETS_FOLDER_PATH = './Datasets/'

# Unimos la carpeta de destino con el nombre de cada archivo
DATASETS_PATH = [path_concat(DATASETS_FOLDER_PATH, url.split('/')[-1][:-3]) for url in DATASETS_URL]

# Ruta del archivo para el muestreo
MUESTRA_DS_PATH = './Datasets/muestra.jsonl'

# Ruta del dataframe final del EDA con sentimientos en OneHot
CODIFIED_DF_PATH = path_concat(DATASETS_FOLDER_PATH, 'codified_df.csv')

# Ruta del dataframe final del procesamiento con procesado
PROCESSED_DF_PATH = path_concat(DATASETS_FOLDER_PATH, 'processed_df.csv')

