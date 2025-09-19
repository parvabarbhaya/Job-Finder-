from flask import Flask , render_template, request ,redirect,url_for
from utils import fetch_news , Lematizer_function,text_preprocessing,hedlines_preprocessing ,data_deployment , fetch_data_from_db
import pickle
import os 
from dotenv import load_dotenv
load_dotenv()

# Load the model
# vectors_path = os.getenv('vector_data_path')
# sentimental_model_path = os.getenv('sentiment_model_path')


csv_path = os.getenv('path_to_csv_file')



# ------------------------------------------------
with open('sentimet_model.pkl',"rb") as f:
    sentimental_model = pickle.load(f)

# # Load the vectorizer
with open('vector_embeddng.pkl', "rb") as f:
    vectorizer = pickle.load(f)
# ------------------------------------------------------


app = Flask(__name__)

@app.route("/" , methods = ['POST', 'GET'])
def Home_page():
    if request.method =='POST':
        query = request.form['query']
        df = fetch_news(query)

        data_deployment(csv_path,text_preprocessing,vectorizer ,sentimental_model)
        return redirect(url_for('data_showing_func'))
    return render_template('Home_page.html')

# ------------------------------------------------

@app.route("/data")
def data_showing_func():
    

    rows = fetch_data_from_db()
    return render_template('news.html' , rows = rows)

# corpus = text_preprocessing(data['Description'],Lematizer_function)

if __name__ == "__main__":
    app.run(debug = True )