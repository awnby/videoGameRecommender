# videoGameRecommender
Hello! This is a video game recommendation system that I worked on with two of my university friends early in 2023.

Having taken a Data Mining course together, we were excited at the thought of building our own recommendation system, given how important they are to our everyday lives. 
Wherever you go, companies spend a huge chunk of their money dedicated to not only advertising their product to the general public, but rather catering their advertisements
to the consumers who would be the most interested. Essentially, almost everything you see has been, in some way, chosen for you specifically to see. Scary, but exciting!
So, when we came up with the idea to work on this small project, we were stumped on what recommender system to build. Well, what better domain to work in than the one you
already know best: video games.

Despite what your parents tell you, video games are crucial to society, and won't be going away anytime soon (take that, Mom!). But gone are the days of getting an ad on TV to 
play a new game that you beg your parents to buy for you, or even walking blindly into a GameStop and picking the game that speaks to you the most (in this case, judging a book
by its cover is valid...). Nowadays, we have become more accustomed to allowing the algorithm dictate what we see on a daily basis. I, personally, no longer search for games on 
Steam but rather scroll mindlessly down the endless page of recommendations that Valve has to offer.

But there's one problem. I don't think their recommendation system really works that well. Or at least for me. So, let's try and make one ourselves.

In order to do so, we will be using two datasets: the IMDB dataset of video games AND the Steam dataset of user activity. The former has a primary key of video games registered
on IMDB, with their genres and ratings and such, while the latter has a primary key of users and their playtime for a whole host of games. Each dataset has over 10,000 entries! So, 
we thought this would be perfect for this project.

Video Game Recommendation Techniques

# ABSTRACT
The purpose was to be able to create a recommendation system for Steam video games based on various genres and the hours played by different users. There were two datasets utilized from Kaggle.com, the first one being about IMDB games and the other about Steam games. The paper observes the effect of two different techniques, with cosine similarity as the baseline and singular value decomposition to build off it to see how well it can recommend games based on certain attributes. Through various features such as different games and user data, the results were visualized through both a heatmap and similarity matrix.

# INTRODUCTION
Video games have become a prevalent part of our culture as one of the most popular forms of entertainment. Within this industry, new video games have constantly been pushed out in order to maintain the interests of the users. This has led to millions of video games with a wide assortment of topics being made available. With this in mind, a method to figure out how to recommend new video games among these millions of different games was to be explored. 
Steam has been widely regarded as one of the more popular game platforms that hosts thousands of games and allows for updates and purchases all in one place. With the wide variety of games and millions of players, we chose to focus on just Steam games for our recommendation system. 
There were two data sets that were used both about video games and from Kaggle, the first one was a Steam dataset that had about 11,000 users with various playtimes for different Steam games, while the second was from IMDB which included 20,803 entries about a wide variety of genres for each game. The main objective was to combine features from both of these data sets such as the user play time and genres for each game in order to recommend similar games based on an input. The approach in order to accomplish this included: Exploratory Data Analysis, Preprocessing, and Analysis. 
Exploratory data analysis allows for initial investigations on the data to get a sense of how it is formatted (outliers and patterns) through statistical summaries. Data preprocessing is a method in which we clean/manipulate the data in order to utilize it for analysis. For the analysis portion of the project, we applied cosine similarity and singular value decomposition. 

# EXPLORATORY DATA ANALYSIS 
The first part of our project was to do some analysis on how the data was formatted from both of the sources. After doing a basic summary, it was shown that none of the columns had null values and what the data type of each of the columns were. To explore the data, we chose to use pandas to get it in a dataframe from a csv. 




Figure 1: Summary of Steam and IMDB dataset columns and their attributes 


Figure 2: Graphical summary of genres for all of the different games 

Figure 3: Numerical summary of genres for all of the different games 
After getting a graph of the different genres of the games, there definitely were a lot more games for certain genres such as crime and thriller than adventure and action which could’ve perhaps led to varying results. 
3 PREPROCESSING 
The goal of preprocessing was to get it into a format that could be best used for analysis. We decided to try and make a dataframe that would combine both data sets and be used by games with genres on the end and standardized hours in each cell. 
3.1 Data cleaning 
The first step in the preprocessing step was to clean the data for any unnecessary variables and rows. From the first data set, the IMDB one, there were a lot of columns that we were not interested in and chose to only keep the columns that were related to the genre of each of the games. After dropping the columns for the IMDB dataset we were left with the games and the genres: Action, Adventure, Comedy, Crime, Family, Fantasy, Mystery, Sci-fi, and Thriller. In the Steam dataset, we dropped the rows that contained the word purchase because it meant the game did not have any hours played. Since there were no null values discovered in the exploratory data analysis, there was nothing to change. 

3.2 Combining Data
To get all the data into a format to process, we had to find a way to put everything into a singular dataframe. By using all the users from the Steam data set with the games as the columns, and hours played in each cell, we were able to add on all the genres to the end of the dataframe and input the hours played for each genre by utilizing a dictionary that contained the game names as keys and the genres as the values. However, since there were a lot of games that were not played by all users, we reduced all of the null values to 0. There were also rows with all 0 since some of the games were not present in both datasets. We also chose to get rid of the rows that contained players who had played 2 or less games because they were not very significant. By adding all of the genres as well, we had to standardize the numbers of hours played separately for the games and genres. We chose to use a centering method around the mean for each two respective sets to standardize the data.

	standardized_value =  (value - mean) / (max - min)


Figure 3.1 : Dataframe after preprocessing (users by games and genres) 
4 ANALYSIS 
The final step in our process was to apply two different algorithms on the processed data set that would be able to recommend the games. Our first baseline model we chose was cosine similarity while the more advanced one we used to build off of was singular value decomposition. 
4.1 Cosine Similarity 
For our baseline model, we chose to use cosine similarity which measures the cosine of the angle between two vectors and helps to determine how similar 2 things may be. This is often used in recommendation systems, especially in user based collaborative filtering. Below is the cosine similarity matrix that we ended up with.


Figure 4.1: Similarity matrix of correlation between different users 
Figure 4.2: Dataframe of each user’s top three most similar users  
Using this matrix, we are now able to compile a list of any given user’s top three “most similar” users, which we thought to create a dataframe out of, as seen to the right.
    With this, we may now build a small recommendation program, which will take an inputted user, and output a list of games for them to play. This is done by iterating through the input’s corresponding tuple in Figure 4.2, and grabbing a sample of each of their most similar users’ most played games. After filtering out any game that is also found in the top five most played games of the input, the program now has a list of recommendations.
   While not an entirely foolproof approach, as the filtering system may miss some repeats or sequels to a game (which doesn’t provide much valuable information to the input user), the program does perform quite well. 
  For example, we ran it on a randomly selected user, whose favorite games included “Sid Meier’s Civilization V,” “TF2,” and “Sniper Elite 3.” With a bit of research, we notice that this particular user enjoys both strategy games and shooters. The program then outputted “Might & Magic Heroes VI” and “CS:GO,” just to name a few. Interestingly enough, “Might & Magic Heroes VI” is a real-time strategy game (much like “Civilization”) and “CS:GO” is a team-based tactical first person shooter (much like “TF2”). Most surprisingly, however, was the program’s most recommended game, “XCOM Enemy Unknown,” which happens to include major elements of real-time strategy and shooters. So, while imperfect, the cosine similarity program seems to be doing an impressive job in their own right.
4.2 Singular Value Decomposition 
SVD helps find the optimal set of factors that best predict the outcome and helps reduce dimensionality of data to find connections and make a recommendation system. It is a matrix factorization technique and an item based collaborative filtering model. 

The justification for the use of SVD was because it is commonly used for item-based collaborative filtering. By observing the similarities between games, we are able to make our recommendation using one game and grabbing the most similar games. On top of that, it adds an extra layer for our recommender to be more accurate as the Cosine Similarity Algorithm is a user-based collaborative filtering algorithm, thus we can compile the results of the two algorithms to work in tandem with one another. 

With the data frame created from the SVD, not only do we have data on the similarities of all the games to one another, but we can also input a singular game and get the top results in terms of similarities. 

In order to accomplish all of this, we utilized sklearn to perform our analysis with the truncated SVD function. Then to visualize our results, we used the correlation matrix from the SVD to create a correlation matrix to see how much each game was related to each other as depicted in figure 4.3 and a closer up version of a sample of 20x20 games in figure 4.4. 

A simple sample run of the program on the game “8BitMMO” returns “Scribblenauts Unlimited” as the most similar game. This is an appropriate recommendation as both games are creative sandbox games, so if someone plays “8BitMMO” for a lot of hours, they’re quite likely to like “Scribblenauts Unlimited” as it is similar in terms of the hours spent playing. 

 

Figure 4.3 Correlation matrix of video games 

Figure 4.4 20x20 sample correlation matrix of video games 
5 CONCLUSIONS
The recommendation system that came out of the algorithms are observed to make effective game recommendations from the database. From a rudimentary observation comparing the genres of the input games and the output recommended games, we can see that it is effectively recommending games with similar genres.

One consideration to note for our project is we did not use quantitative accuracy measures. Due to the fact that we created a recommender system, typical data mining practices of quantitative accuracy are not appropriate to evaluate whether or not it’s a good recommendation. For recommender systems, the effectiveness can only be measured by individual opinion as whether or not something is a “good” recommendation is entirely subjective.

For further study, k-nearest neighbors can be applied to the dataset to create an extra filter that serves as a Clustering Based Recommendation System to supplement the Collaborative Based Filtering methods in this project. In addition, we should swap the iMDB dataset we used for a Steam specific dataset so we can keep more data points than what was observed in the present study.
ACKNOWLEDGMENTS
We would like to thank Prof. T and all of the TAs for all of the support they have given us throughout the course. 

REFERENCES
[1] Antonov, A. (2019, June 16). Steam games complete dataset. Kaggle. Retrieved April9,2023,fromhttps://www.kaggle.com/datasets/trolukovich/steam-games-complete-dataset 
[2] GeeksforGeeks. (2023, January 16). Singular value decomposition (SVD). GeeksforGeeks.RetrievedApril9,2023,fromhttps://www.geeksforgeeks.org/singular-value-decomposition-svd/ 
[3] Sierram. (2017, July 9). Cosine similarity wine descriptions. Kaggle. Retrieved April9,2023,fromhttps://www.kaggle.com/code/sierram/cosine-similarity-wine-descriptions 
[4] Talay, M. A. (2022, September 2). IMDB video games. Kaggle. Retrieved April 9,2023,fromhttps://www.kaggle.com/datasets/muhammadadiltalay/imdb-video-gmes 
[5] Tam, A. (2021, October 28). Using singular value decomposition to build a recommender system. MachineLearningMastery.com. Retrieved April 9, 2023, fromhttps://machinelearningmastery.com/using-singular-value-decomposition-to-build-a-recommender-system/ 

APPENDIX

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import *
import sys
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
## EDA
imdb = pd.read_csv("imdb-videogames.csv")
imdb = imdb.drop(imdb.columns[[0, 2, 3, 4, 5, 6, 7]],axis = 1)

steam = pd.read_csv("steam-200k.csv", header=None)
steam = steam[steam.iloc[:, -3].str.contains("purchase") == False]
steam = steam.drop(steam.columns[[2, 4]], axis = 1)
steam.columns = ["Player", "Game", "Playtime"]

print(imdb.head())
print(steam.head())
print("IMDB Summary Statistics")
imdb.describe()
import matplotlib as plt
from matplotlib import pyplot as plt

freq_imdb = pd.DataFrame({'Genre': ['Action', 'Adventure', 'Comedy', 'Crime', 'Family', 'Fantasy', 'Sci-Fi', 'Mystery', 'Thriller'],
                          'Frequency': [11930, 10469, 19020, 19908, 17928, 16453, 19618, 17213, 20280]})

plt.plot(freq_imdb['Genre'], freq_imdb['Frequency'])
plt.show()
# creating dictionary of games and genres
dict1 = {}
for index, row in imdb.iterrows():
    if row["name"] not in dict1: 
        dict1[row["name"]] = []
        if row["Action"] == True:
            dict1[row["name"]].append("Action")
        if row['Adventure'] == True:
            dict1[row['name']].append("Adventure")
        if row['Comedy'] == True:
            dict1[row['name']].append("Comedy")
        if row['Crime'] == True:
            dict1[row['name']].append("Crime")
        if row['Family'] == True:
            dict1[row['name']].append("Family")
        if row['Fantasy'] == True:
            dict1[row['name']].append("Fantasy")
        if row['Sci-Fi'] == True:
            dict1[row['name']].append("Sci-Fi")
        if row['Mystery'] == True:
            dict1[row['name']].append("Mystery")
        if row['Thriller'] == True:
            dict1[row['name']].append("Thriller")
df2 = pd.DataFrame({'user': steam.iloc[:,0], 'games': steam.iloc[:,1],'hours': steam.iloc[:,2]})
df2 = df2.pivot_table(index='user', columns='games', values='hours')
df3 = df2.reindex(columns = df2.columns.tolist() + ['Action', 'Adventure', 'Comedy', 'Crime', 'Family', 'Fantasy', 'Sci-Fi', 'Mystery', 'Thriller'])
df3 = df3.fillna(0)
#adding in hrs per catergory to df
for index, row in df3.iterrows():    
    for i in range(len(df3.columns)):
        if row[i] > 0:
            if df3.columns[i] in dict1:
                for j in dict1[df3.columns[i]]:
                    row[j] += row[i]
df3.head()
df3 = df3.loc[(df3[['Action', "Adventure", "Comedy", "Crime", "Family", "Fantasy", "Sci-Fi", "Mystery", "Thriller"]] != 0).all(axis=1)]
df_games = df3.iloc[:,:-9]
df_genres = df3.iloc[:,-9:]

print("Full Dataframe Summary Statistics")
df3.describe()
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

def center(row):
    new_row = (row - row.mean()) / (row.max() - row.min())
    return new_row

df_games_std = df_games.apply(center, axis = 1)
df_genres_std = df_genres.apply(center, axis = 1)
df_genres_std
frames = [df_games_std, df_genres_std]
  
df_joined_std = pd.concat([df_games_std, df_genres_std], axis=1, join='inner')
df_joined_std.head()
print("Standardized Full Dataframe Summary Statistics")
df_joined_std.describe()
cosine_sim_matrix = pd.DataFrame(cosine_similarity(df_joined_std), columns = df_joined_std.T.columns)

print(cosine_sim_matrix)
for i in range(len(cosine_sim_matrix.columns)):
    cosine_sim_matrix = cosine_sim_matrix.rename(columns={cosine_sim_matrix.columns[i]: str(i)})

print(cosine_sim_matrix)
top_three_sim_genres = pd.DataFrame(cosine_sim_matrix.columns.values[np.argsort(-cosine_sim_matrix.values, axis=1)[:, 1:4]], 
                  index=cosine_sim_matrix.index,
                  columns = ['1st Max','2nd Max','3rd Max']).reset_index().drop(['index'], axis = 1)
print(top_three_sim_genres)   
top_three_1 = df_games.iloc[46,:].nlargest(5).index.tolist()
print(top_three_1)
def cosine_recommender(user, dataframe):
    full_list = []
    for j in range(3):
        i = 1
        top_three = []
        user_top_three = dataframe.iloc[user,:].nlargest(3).index.tolist()
        while len(set(top_three) - set(user_top_three)) < 3:
            top_three = dataframe.iloc[int(top_three_sim_genres.iloc[user, j]), :].nlargest(i).index.tolist()
            i = i + 1
        for game in top_three:
            if game not in (full_list) and game not in (user_top_three):
                full_list.append(game)
    return full_list

print(cosine_recommender(46, df_games))
X = df3.values.T
SVD = TruncatedSVD(n_components=12, random_state=17)
result_matrix = SVD.fit_transform(X)
result_matrix.shape
corr_matrix = np.corrcoef(result_matrix)
corr_matrix.shape
game_names = df3.columns
games_list = list(game_names)

df_svd = pd.DataFrame(index=list(df3.columns.values))

for column in df3:
    game = games_list.index(column)
    corr_game = corr_matrix[game]
    df_svd[column] = corr_game.tolist()

df_svd
df_svd = df_svd.dropna(how='all')
df_svd = df_svd.dropna(axis=1, how='all')
df_svd
fig = plt.figure(figsize=(48,48), dpi = 480)
sns.heatmap(df_svd,cmap="PiYG",annot=False,square=True,linewidth=0.0001,linecolor="#222")
fig = plt.figure(figsize=(18,18), dpi = 480)
sns.heatmap(df_svd.iloc[0:20,0:20],cmap="PiYG",annot=False,square=True,linewidth=.5,linecolor="#222")

game_index = games_list.index('8BitMMO')

corr_game = corr_matrix[game_index]

fourth = fifth = sixth = third = first = second = -sys.maxsize

for i in range(0, len(corr_game)):

        if (corr_game[i] == 1):
             continue
        
        elif (corr_game[i] > first):
            
            sixth = fifth
            fifth = fourth
            fourth = third
            third = second
            second = first
            first = corr_game[i]
         
        elif (corr_game[i] > second):
            sixth = fifth
            fifth = fourth
            fourth = third
            third = second
            second = corr_game[i]
         
        elif (corr_game[i] > third):
            sixth = fifth
            fifth = fourth
            fourth = third
            third = corr_game[i]

        elif (corr_game[i] > fourth):
            sixth = fifth
            fifth = fourth
            fourth = corr_game[i]

        elif (corr_game[i] > fifth):
            sixth = fifth
            fifth = corr_game[i]

        elif (corr_game[i] > sixth):
            sixth = corr_game[i]

rec = [game_names[(corr_game == first)][0], game_names[(corr_game == second)][0], game_names[(corr_game == third)][0], game_names[(corr_game == fourth)][0], game_names[(corr_game == fifth)][0], game_names[(corr_game == sixth)][0]]
rec


