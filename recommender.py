import pandas as pd
import numpy as np
import random

from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

class Recommender:
    '''
    
    class to recommend k number of movies
    
    '''
    def __init__(self, movies, ratings, k, query):
        '''
        
        class elements:
            -movies dataframe
            -ratings dataframe
            -number of movie to recommend
            -user ratings
            
        '''
        self.movies = movies
        self.ratings = ratings
        self.k = k
        self.query = query

    def set_df(self):
        '''
        
        read movies and ratings dataframes, 
        merge them on movieId column,
        reshape the dataframe:
            -movies titles as columns
            -users Id as index
            -ratings as values
        
        '''
        self.movies_df = pd.read_csv(self.movies)
        self.ratings_df = pd.read_csv(self.ratings)

        self.merged_df = pd.merge(self.movies_df, 
                                  self.ratings_df, 
                                  on = 'movieId')
        
        self.initial_df = pd.pivot_table(self.merged_df, 
                                         values = 'rating', 
                                         index = 'userId', 
                                         columns = 'title')
        
    def cleaned(self):
        '''
        
        clean the initial dataframe and replace null values with mean
        
        call set_df() function
        
        '''
        self.set_df()
        '''
        
        count not null values, 
        create a list of columns with less than 50 not null values, 
        drop that columns from the dataframe
        
        calculate the mean value of all ratings, 
        use mean to replace all null values
        
        '''
        self.count = self.initial_df.notna().sum()
        self.cols_to_drop = list(self.count[self.count < 50].index)
        self.column_removed_df = self.initial_df.drop(self.cols_to_drop, 
                                                      axis = 1)
        
        self.col_mean = self.column_removed_df.mean()
        self.cleaned_df = self.column_removed_df.fillna(value = self.col_mean)

    def cleaned_cosine(self):
        '''
        
        clean the initial dataframe and replace null values with 0
        
        call set_df() function
        
        '''
        self.set_df()
        '''
        
        count not null values, 
        create a list of columns with less than 50 not null values, 
        drop that columns from the dataframe
        
        replace null values with 0
        
        '''
        self.count = self.initial_df.notna().sum()
        self.cols_to_drop = list(self.count[self.count < 50].index)
        self.column_removed_df = self.initial_df.drop(self.cols_to_drop, 
                                                      axis = 1)
        
        self.cleaned_df = self.column_removed_df.fillna(value = 0)
    
    def factorizer(self):
        '''
        
        define and train a non-negative matrix factorization model
        
        call cleaned() function
        
        '''
        self.cleaned()
        '''
        
        create an instance of the NMF model, 
        unsupervised learning algorithm that factorize a non negative matrix in two non negative matrixes, 
        the goal is to approximate the first matrix by the product of the second two.
        
        n_components define the number of rows in a matrix and column in the other, 
        max_iter define the max number of iterations to perform before stopping the algorithm.
        
        fit() this model on cleaned_df(the dataset with mean inplace of null values)
        
        '''
        self.factorizer_model = NMF(n_components = 30, 
                                    max_iter = 1000)
        self.factorizer_model.fit(self.cleaned_df)

    def matrixes(self):
        '''
        
        calculate two non negetive matrixes and multiply them, 
        approximate the first dataframe by the product of two non negative matrixes, 
        Q * P = R_hat
        
        call cleaned() function
        call factorizer() function
        
        '''
        self.cleaned()
        self.factorizer()
        '''
        
        define a list of movies from column names, 
        define a list of user from index names
        
        define Q matrix with model.components_, 
        convert it to a pandas dataframe with movies title as column names
        
        define P matrix with .transform(df), 
        convert it to a pandas dataframe with users Id as index
        
        define R_hat with np.dot(matrix moltiplication), 
        convert it to a pandas dataframe with users Id as index and movies titles as column names
        
        '''
        self.movies_list = list(self.cleaned_df.columns)
        self.users_list = list(self.cleaned_df.index)

        self.Q = self.factorizer_model.components_
        self.Q = pd.DataFrame(self.Q, 
                              columns = self.movies_list)
        
        self.P = self.factorizer_model.transform(self.cleaned_df)
        self.P = pd.DataFrame(self.P, 
                              index = self.users_list)
        
        self.R_hat = np.dot(self.P, 
                            self.Q)
        self.R_hat = pd.DataFrame(self.R_hat, 
                                  index = self.users_list, 
                                  columns = self.movies_list)
        
    def user_random_ratings(self):
        '''
        
        calculate random ratings for random movies, 
        both for the cleaned dataframe and for the initial dataframe
        
        call cleaned() function
        call set_df() function
        
        '''
        self.cleaned()
        self.set_df()
        '''
        
        list of 3 random movie titles from cleaned df,
        for every movie in that list append to another list a random rating between 1-5, 
        create a dictionary with movie titles as keys and respective random rating as values
        
        list of 3 random movie titles from initial df, 
        for every movie in that list append to another list a random rating between 1-5, 
        create a dictionary with movie titles as keys and respective random rating as values
        
        '''
        self.random_movies_clean = random.sample(list(self.cleaned_df.columns), 
                                                 5)
        self.random_rating_clean = []
        for i in self.random_movies_clean:
            self.rating_clean = random.randint(1, 
                                               5)
            self.random_rating_clean.append(self.rating_clean)
        self.user_random_ratings_clean = dict(zip(self.random_movies_clean, 
                                                  self.random_rating_clean))
        
        self.random_movies_initial = random.sample(list(self.initial_df.columns), 
                                                   3)
        self.random_rating_initial = []
        for i in self.random_movies_initial:
            self.rating_initial = random.randint(1, 
                                                 5)
            self.random_rating_initial.append(self.rating_initial)
        self.user_random_ratings_initial = dict(zip(self.random_movies_initial, 
                                                    self.random_rating_initial))
        
    def random_movies(self):
        '''
        
        function to return a list of 5 random movies
        
        call user_random_ratings() function
        
        '''
        self.user_random_ratings()

        return self.random_movies_clean
        
    def recommender_NMF(self):
        '''
        
        recommend k number of movies from users input using a non-negative matrix factorization model
        
        call cleaned() function
        call matrixes() function
        call factorizer() function
        
        '''
        self.cleaned()
        self.matrixes()
        self.factorizer()
        '''
        
        for loop to replace 0 with mean from user input.
        
        create an empty recommendations list.
        
        convert user input into pandas dataframe,
        'new_user' as index, 
        movie titles(column names) as columns,
        fill null values with mean.
        
        create P matrix with .transform() from user ratings dataframe,
        convert this matrix into a pandas dataframe, 
        'new_user' as index.
        
        calculate R_hat for user multiplying P_user and Q matrixes, 
        convert R_user_hat into a pandas dataframe, 
        movie names as columns, 
        'new_user' as index, 
        transpose rows and columns of the dataframe, 
        sort values in descending order by the values in the column labeled 'new_user'.
        
        create a list with the movies voted by the user, 
        create a list with the movies from the first to the last recommandables, 
        loop through the recommandables movies excluding the movies voted by the user,
        append the result to the recommandations list, 
        create a movies_to_recommend list with the first 10 movies of recommendations list.
        
        '''
        for value in self.query:
            if value == 0:
                self.query[value] = self.col_mean
        
        self.recommendations = []

        self.user_input = pd.DataFrame(self.query, 
                                       index = ['new_user'], 
                                       columns = self.cleaned_df.columns)
        self.user_input_imputed = self.user_input.fillna(value = self.cleaned_df.mean())

        self.P_user = self.factorizer_model.transform(self.user_input_imputed)
        self.P_user = pd.DataFrame(self.P_user, 
                                   index = ['new_user'])
        
        self.R_user_hat = np.dot(self.P_user, self.Q)
        self.R_user_hat = pd.DataFrame(self.R_user_hat, 
                                       columns = self.cleaned_df.columns, 
                                       index = ['new_user'])
        self.R_user_hat_transposed = self.R_user_hat.T.sort_values(by = 'new_user', 
                                                                   ascending = False)
        
        self.query_list = list(self.query.keys())
        self.recomandables = list(self.R_user_hat_transposed.index)
        for m in self.recomandables:
            if m not in self.query_list:
                self.recommendations.append(m)
        self.movies_to_recommend = self.recommendations[0 : self.k]

    def show_nmf(self):
        '''
        
        function to return movies_to_recommend list
        
        call recommender_NMF() function
        
        '''
        self.recommender_NMF()
        '''
        
        return the list with the 10 movies to recommend defined with the recommender_NMF() function
        
        '''
        print(self.query)
        return self.movies_to_recommend
    
    def recommender_cosine(self):
        '''
        
        define 10 movies to recommend with cosine similarity.
        
        call cleaned_cosine() function.
        
        '''
        self.set_df()
        self.cleaned_cosine()
        '''
        
        create a pandas dataframe with the user ratings, 
        'new_user' as index,
        movie titles as columns, 
        append the user input dataframe to cleaned_df.

        define new_user variable with 'new_user' string, 
        transpose the dataframe, 
        fill null values with zeros.
        
        calculate cosine similarity between users, 
        round to the 2 decimal number, 
        convert this matrix to a pandas dataframe with user Id as index and as columns.
        
        define a list of movies unseen by the user, 
        define a list of users from the most to the less similar to the new_user.
        
        initialize an empty dictionary,
        loop through the movies not seen by the user, 
        define the movies not rated by users, 
        convert this list to a set.
        
        define num as 0, 
        define den as 0.
        
        loop through the intersection between the set of movies not rated and the users from the more to the less similar, 
        store the rating that user u give to movie m,
        store the similarities between new_user and another user who have rated the same movie.

        define num as the sum between himself and the product of the rating that user u give to movie m and the similarities between new_user and another user who rateed the same movie,
        define den as the sum between himself and the rating that user u give to movie m, himself and 0,0001.

        define ration as the division between num and den, 
        append ratio as value of cosine_dict for every key = m.
        
        sort cosine_dict using the sorted() function with the key parameter set to a lambda function that return the second item of every key value pair, 
        create a list from keys of cosine_dict and extract the first 10 movies.
        
        '''
        self.user_input_cosine = pd.DataFrame(self.query, 
                                              index = ['new_user'], 
                                              columns = self.cleaned_df.columns)
        self.cleaned_df = self.cleaned_df.append(self.user_input_cosine)

        self.new_user = 'new_user'
        self.cleaned_df = self.cleaned_df.T
        self.user_item = self.cleaned_df.fillna(value = 0)

        self.user_user_matrix = cosine_similarity(self.user_item.T).round(2)
        self.user_user_matrix = pd.DataFrame(self.user_user_matrix, 
                                             columns = self.user_item.columns, 
                                             index = self.user_item.columns)
        
        self.unseen_movies = self.cleaned_df[self.cleaned_df['new_user'].isna()].index
        self.top_users = self.user_user_matrix['new_user'].sort_values(ascending= False).index

        self.cosine_dict = {}
        for m in self.unseen_movies:
            self.other_users = self.cleaned_df.columns[~self.cleaned_df.loc[m].isna()]
            self.other_users = set(self.other_users)

            self.num = 0
            self.den = 0

            for u in self.other_users.intersection(set(self.top_users)):
                self.rating = self.user_item[u][m]
                self.sim = self.user_user_matrix[self.new_user][u]

                self.num = self.num + (self.rating * self.sim)
                self.den = self.den + self.sim + 0.0001

            self.ratio = self.num / self.den
            self.cosine_dict[m] = [self.ratio]

        self.cosine_dict = dict(sorted(self.cosine_dict.items(), 
                                       key = lambda x: x[1], 
                                       reverse = True))
        self.ten_movies = list(self.cosine_dict.keys())
        self.ten_movies = self.ten_movies[:10]

    def show_cosine(self):
        '''
        
        function to return the list of 10 movies to recommend defined with recommender_cosine()
        
        call recommender_cosine()
        
        '''
        print(self.query)
        self.recommender_cosine()

        return self.ten_movies
    
    def recommender_random(self):
        '''
        
        function to recommend 10 favourite movie of a random user who rates 5 random movies with random values between 0-5
        
        call cleaned() function
        call matrixes() function
        call user_random_ratings() function
        call factorizer() function
        
        '''
        self.cleaned()
        self.matrixes()
        self.user_random_ratings()
        self.factorizer()
        '''
        
        create an empty reccomandations list,
        define user_input dataframe with 5 random ratings of 5 random movies,
        new_user as index and movies names as column names.
        
        fill null values with mean,
        transform the dataframe with transform() function of the factorizer model, 
        convert the dataframe with the transformed data into a pandas dataframe with new_user as index.
        
        apply a matrix moltiplication between P_user matrix and Q matrix, 
        convert the result into a pandas dataframe with movie names as columns and new_user as index, 
        transpose the dataframe and sort his values by new_user in descending order.
        
        define a list of the movies rated randomly, 
        define a list of recommandables movies converting r_hat dataframe's index into a list.
        
        loop through every recommandables movies that has not been rated by the random user, 
        append them in recommandations list, 
        define a list with the first k films in recommandations list.
        
        '''
        self.recommendations = []
        self.user_input = pd.DataFrame(self.user_random_ratings_clean, 
                                       index = ['new_user'], 
                                       columns = self.cleaned_df.columns)
        
        self.user_input_imputed = self.user_input.fillna(value = self.cleaned_df.mean())
        self.P_user = self.factorizer_model.transform(self.user_input_imputed)
        self.P_user = pd.DataFrame(self.P_user, 
                                   index = ['new_user'])
        
        self.R_user_hat = np.dot(self.P_user, self.Q)
        self.R_user_hat = pd.DataFrame(self.R_user_hat, 
                                       columns = self.cleaned_df.columns, 
                                       index = ['new_user'])
        self.R_user_hat_transposed = self.R_user_hat.T.sort_values(by = 'new_user', 
                                                                   ascending = False)
        
        self.random_ratings_clean_list = list(self.user_random_ratings_clean.keys())
        self.recomandables = list(self.R_user_hat_transposed.index)

        for m in self.recomandables:
            if m not in self.random_ratings_clean_list:
                self.recommendations.append(m)
        self.movies_to_recommend = self.recommendations[0 : self.k]
    
    def show_random(self):
        '''
        
        function to return the list of k movies to recommend by random ratings of random movies by random user.
        
        call recommender_random() function
        
        '''
        self.recommender_random()
        return self.movies_to_recommend