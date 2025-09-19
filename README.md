📰 News Sentiment Analysis with Flask
<!-- ####################################################### -->
Hey! Welcome to my final project. This is a simple web app that does sentiment analysis on news articles. It's built with Flask and uses a basic Naive Bayes model to figure out if a news story is Positive, Negative, or Neutral. I'm pretty stoked about it because it's a good first step into building real-world ML applications.

How it Works
The whole thing is pretty straightforward. You just type a search query on the website's homepage, and it does a bunch of stuff behind the scenes:

-> It uses the NewsAPI to grab the latest headlines and articles based on your search.

-> Then, it cleans up the text, getting rid of all the extra noise.

-> The cleaned text goes into a pre-trained Naive Bayes model which I trained on some news data.

-> After the model makes its predictions, it saves everything—the headlines, the text, and the predicted sentiment—into an SQLite database.

-> Finally, it shows you all the results on a separate page.

I know the model's accuracy is around 60% right now, which isn't the best, but it's a good starting point and a great learning experience! It shows that even with a simple model, you can still get some decent results.

Technologies Used
Here are the main libraries and tools I used to build this:

1. python

2. Flask: For the web application framework.

3. scikit-learn: For the Naive Bayes model and the TF-IDF vectorizer.

4. NLTK: For all the text preprocessing stuff like lemmatization.

5. pandas: To handle and manage the data.

6. requests: To pull data from the NewsAPI.

7. sqlite3: To store the results in a database.

8. python-dotenv: To manage the API key.

