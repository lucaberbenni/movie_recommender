from flask import Flask, render_template, request
from recommender import Recommender

app = Flask(__name__)

@app.route('/')
def homepage():
    user_query = request.args.to_dict()
    user_query = {key : int(value) for key, value in user_query.items()}
    recommender = Recommender(movies = 'data/movies.csv', 
                          ratings = 'data/ratings.csv', 
                          k = 10,  
                          query = user_query)
    return render_template('home.html', movies = recommender.random_movies())

@app.route('/results')
def rec():
    user_query = request.args.to_dict()
    user_query = {key : int(value) for key, value in user_query.items()}
    recommender = Recommender(movies = 'data/movies.csv', 
                          ratings = 'data/ratings.csv', 
                          k = 10,  
                          query = user_query)
    return render_template('results.html', movies = recommender.show_nmf(), random = recommender.random_movies())

@app.route('/random')
def rec_random():
    user_query = request.args.to_dict()
    user_query = {key : int(value) for key, value in user_query.items()}
    recommender = Recommender(movies = 'data/movies.csv', 
                          ratings = 'data/ratings.csv', 
                          k = 10,  
                          query = user_query)
    return render_template('random.html', movies = recommender.show_random())

@app.route('/cosine_results')
def rec_cosine():
    user_query = request.args.to_dict()
    user_query = {key : int(value) for key, value in user_query.items()}
    recommender = Recommender(movies = 'data/movies.csv', 
                          ratings = 'data/ratings.csv', 
                          k = 10,  
                          query = user_query)
    return render_template('cosine_results.html', movies = recommender.show_cosine())

if __name__ == '__main__':
    app.run(port = 5000, 
            debug = True)
