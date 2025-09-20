
import pandas as pd
import requests
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()


NEWS_API_KEY = os.getenv('news_api')

def fetch_news(query,page_size=10):

    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    
    data = response.json()
    articles = data["articles"]
    df = pd.DataFrame([{
            'Heading':art.get('title'),
            'Context': (art.get('description')+ (art.get('content'))),
            'Link': (art.get('url'))

        }for art in articles])
    df.to_csv("news_data.csv",mode="a",index=False,header=not os.path.exists('news_data.csv'))
    return df




#   --------------------------------------------------------------
# preprocessing the data 
# preprocesing P1 Textpreprocessing , Lower cace , and implementing regex on whole text . 
import pandas as pd
import nltk
# from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re 
from nltk.stem import WordNetLemmatizer



from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.svm import SVC
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

def Lematizer_function (Text):

    lematizer = WordNetLemmatizer()
    return lematizer.lemmatize(Text)
# --------------------------------------------------------
def text_preprocessing (text , Lematizer_function):
    """ this funtion help to preprocess the data 
        1.Lowercasing
        2.Removing non-alphabets
        3.Stopword removal
        4.Tokenition
        5.Lemmatization
        6.storing the clean text in to list and return the list 
    
    """
    stop_words = set(stopwords.words('english'))
    corpus  = []
    for i in text:
        try:
            # Text lowering
            text = i.lower()

            # removing non-alphabatic characxters 
            text = re.sub(r'[^a-zA-Z]',' ',text)
            # Tokenization of words 
            words_tokens = word_tokenize(text,language="english")
            # remove stop_words and apply Lematization
            clean_word =  [Lematizer_function (word) for word in words_tokens if  not word  in stop_words]

            # join the clean word to make sentence 
            clean_text = ' '.join(clean_word)

            # append the clean word to corpus list 

            corpus.append(clean_text)



        except Exception as e:
            print("Erroe:",e)
            return corpus.append("")
    return corpus
# ------------------------------------------------------
def hedlines_preprocessing (text ):
    hedlines  = []
    for i in text:
        try:

            # removing non-alphabatic characxters 
            text = re.sub(r'[^a-zA-Z]',' ',i)
            cleaned_text = text.strip().lower()


            # append the clean word to hedlines list 

            hedlines.append(cleaned_text if cleaned_text else "")



        except Exception as e:
            print("Erroe:",e)
            return hedlines.append("")
    return hedlines

# deployment code 

# ---------------------------------------------
# deploy the model usigg .env , Lematization, text_preprocessing , Hedlines cleaning , vectorization , and moedel prediction



csv_path = os.getenv('path_to_csv_file')
csv_path
# Load the model

def data_deployment(csv_path ,text_preprocessing, vectorizer, sentimental_model):
    # 1. Load dataset
    raw_data = pd.read_csv(csv_path)

    # hedline _cleaning 
    hedlines = hedlines_preprocessing(raw_data['Heading'])

    # 2. Preprocess text
    preprocessed_data = text_preprocessing(raw_data['Context'], Lematizer_function)

    # 3. Vectorize (DON'T fit again, only transform)
    raw_data_vec = vectorizer.transform(preprocessed_data)

    # 4. Predict
    raw_data_prediction = sentimental_model.predict(raw_data_vec)

    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    # 5. storing it in to the SQLite3 
    connection_obj = sqlite3.connect('News_data.db')
    cursor_obj = connection_obj.cursor()
    cursor_obj.execute("""
    CREATE TABLE IF NOT EXISTS news_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Headline TEXT,
        Sentence TEXT,
        Sentiment TEXT
    )
    """)

    # 6. Insert Records into data.
    for hedline, sentence, pred in zip(hedlines, preprocessed_data, raw_data_prediction):
        cursor_obj.execute(
            "INSERT INTO news_predictions (Headline, Sentence, Sentiment) VALUES (?, ?, ?)",
            (hedline, sentence, sentiment_map[pred])
        )

    connection_obj.commit()
    connection_obj.close()
    
    return f" {hedline}\n , {sentence}\n ,{sentiment_map}"


def fetch_data_from_db(db ='News_data.db'):
    con = sqlite3.connect(db)
    cur = con.cursor()
    statement = '''SELECT * FROM news_predictions'''
    cur.execute(statement)
    output = cur.fetchall()
    con.close()
    return output

