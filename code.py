#import pandas
import pandas as pd
#read the csv training file
train = pd.read_csv("train_E6oV3lV.csv",index_col="id")
#exploratory data analysis
###print("Dataframe shape:",train.shape)
###print("Columns:",train.columns.values)
###print(train.head())

#import BeautifulSoup for data cleaning
from bs4 import BeautifulSoup
#import regexp
import re
#import stopwords from nltk
from nltk.corpus import stopwords

#use WordNetLemmatizer to lemmatize the words
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize_tweets(raw_review):
    lemmatized_tweet=[]
    raw_review = raw_review.split()
    for w in raw_review:
        word = wordnet_lemmatizer.lemmatize(w)
        lemmatized_tweet.append(word)
    return(" ".join(lemmatized_tweet))
###print(lemmatize_tweets("plays in sand and listens"))

#function to segregate only tweets
#defining function separate_tweets() to separate hashtags from main tweet
def separate_tweets(raw_review):
    #changing the whole tweet in lowercase so that hashtags with different cases can match and not be identified as different hashtags
    tweets_only = raw_review.lower()
    #finding all the hashtags
    tweets_only = re.findall("#\w+",tweets_only)
    #returning all hashtags as list
    return(" ".join(tweets_only))
#checking if our function works properly
###print(separate_tweets("#i_am_a_good_boy i ama sick # #i_am_tired ##"))
###print(separate_tweets("i am going to change everything #i ## #i_am ### @i @@ @# #@   @a@ab@abc #iam#i## #... #I_am_Avinesh"))
#it works :)

#stopwords stored in a set so that we dont have to access the nltk corpus everytimr
stops = set(stopwords.words("english"))
#function to separate all the words in the whole tweet including separation of words from their hashes and mentions
def review_tweets(raw_review):
    review_text = BeautifulSoup(raw_review,"html.parser").get_text()
    #utilising only the letters only from the whole tweet
    #this separates the words from their # and @
    letters_only = re.sub("[^a-zA-Z]"," ",review_text)
    #changing the whole tweet in lowercase so that tweets with different cases can match and not be identified as different tweets
    words = letters_only.lower().split()
    #splitting the whole tweet on the basis of white space
    meaningful_words = [w for w in words if w not in stops]
    return(lemmatize_tweets((" ".join(meaningful_words))))
#checking if our function works properly
###print(review_tweets("i am going to change everything #i ## #i_am ### @i @@ @# #@   @a@ab@abc #iam#i## #... plays"))
#it works

#calculating the total size of the array
num_reviews = train.tweet.size
clean_tweets = []
clean_hashtags = []
for i in range(1,num_reviews):
    clean_tweets.append(review_tweets(train.tweet[i]))
    clean_hashtags.append(separate_tweets(train.tweet[i]))
    if(i%3000 == 0):
        print(i," tweets processed. . . . ")
print("processing completed")

#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer1 = CountVectorizer(analyzer="word",
                            tokenizer=None,
                            preprocessor=None,
                            stop_words=None,
                            max_features=10000)
vectorizer2 = CountVectorizer(analyzer="word",
                            tokenizer=None,
                            preprocessor=None,
                            stop_words=None,
                            max_features=1000)
print("fitting tweets to vectorizer")
tweets_data_features = vectorizer1.fit_transform(clean_tweets)
print("fitting hashtags to vectorizer")
hashtags_data_features = vectorizer2.fit_transform(clean_hashtags)
#changing it to numpy array as it is easy to wrok with them
###print("changing all vectors in number form")
tweets_data_features = tweets_data_features.toarray()
hashtag_data_features = hashtags_data_features.toarray()

###print("Tweet bag of words:",tweets_data_features.shape)
###print("Hashtag bag of words",hashtag_data_features.shape)
###print("Feature names:",vectorizer1.get_feature_names())

import numpy as np
tweet_dist = np.sum(tweets_data_features,axis=0)
hashtag_dist = np.sum(hashtag_data_features,axis=0)
###for tag,count in zip(vectorizer1.get_feature_names(),tweet_dist):
###    print(count," : ",tag)
###for tag,count in zip(vectorizer2.get_feature_names(),hashtag_dist):
###    print(count," : ",tag)

###print(type(hashtag_data_features))
###print(type(tweets_data_features))
###print(hashtag_data_features.shape)
###print(tweets_data_features.shape)

#joining the two numpy arrays to perform the fitting together
final_data_features = np.concatenate((tweets_data_features,hashtag_data_features),axis=1)
###print(final_data_features.shape)

#applying RandomForest classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit((final_data_features),train["label"])

#read the test file
test = pd.read_csv("test_tweets_anuFYb8.csv")
num_reviews = len(test["tweet"])
clean_tweet_review = []
clean_hash_review = []
print("cleaning and parsing the data . . . .")
for i in range(num_reviews):
    if((i+1)%3000==0):
        print("working on cleaning the test data")
    temp = test["tweet"][i]
    clean_review = review_tweets(test[temp])
    clean_tweet_review.append(clean_review)
    clean_hash = separate_tweets(temp)
    clean_hash_review.append(clean_hash)
tweet_data_features = vectorizer1.transform(clean_tweet_review)
tweet_data_features = test_data_features.toarray()
hash_data_features = vectorizer2.transform(clean_hash_review)
hash_data_features = hash_data_features.toarray()
result = forest.predict(test_data_features)
#create a DataFrame containing the prediction
output = pd.DataFrame(data={"id":test["id"], "label":result})
#output the total result to a .csv file
output.to_csv("predicted.csv",index=False)
