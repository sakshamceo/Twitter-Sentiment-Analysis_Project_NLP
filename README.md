# Twitter Sentiment Analysis by using Logistic Regressions and Naive Bayes Classification
Hi!
I am Saksham Garg, currently studying in 3rd year and persuing my Bachelor's of technology in Information Technology

## Aim
To Perform sentiment analysis of tweets using logistic regression and then understanding naïve Bayes classification on training our model. [Twitter Sentiment Analysis Project](https://github.com/sakshamceo/Twitter-Sentiment-Analysis_Project_NLP/blob/main/main_project_using_Logistic_Regression.ipynb)

## About
Sentiment Analysis is the process of ‘computationally’ determining whether a piece of writing is positive, negative or neutral. It’s also known as opinion mining, deriving the opinion or attitude of a speaker. 
## Why sentiment analysis? 
1. <strong> Business: </strong> In marketing field companies use it to develop their strategies, to understand customers’ feelings towards products or brand, how people respond to their campaigns or product launches and why consumers don’t buy some
products. <br>
2. <strong> Politics: </strong> In political field, it is used to keep track of political view, to detect consistency and inconsistency between statements and actions at the government level. It can be used to predict election results as well! <br>
3. <strong> Public Actions: </strong> Sentiment analysis also is used to monitor and analyse social phenomena, for the spotting of potentially dangerous situations and determining the general mood of the blogosphere. 
4. Sentiment analysis uses Natural Language Processing (NLP) to make sense of human language, and machine learning to automatically deliver accurate results.

### Building and Visualizing word frequencies
#### Setup
This will build a dictionary where we can lookup how many times a word appears in the lists of positive or negative tweets.
```
import nltk                                 
from nltk.corpus import twitter_samples      
import matplotlib.pyplot as plt              
import numpy as np                            
from nltk.stem import PorterStemmer               
from nltk.corpus import stopwords
```
keys = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
        'song', 'idea', 'power', 'play', 'magnific']
        
        
We will select a set of words that we would like to visualize.

 ![image](https://user-images.githubusercontent.com/59284238/133882745-ff80cb5f-6f80-4df6-988e-06ef00e7dcc9.png) 
  
### Visualizing tweets and the Logistic Regression model

**Objectives**: Visualize and interpret the logistic regression model
#### Setup
```
import nltk                      
from os import getcwd
import pandas as pd           
from nltk.corpus import twitter_samples 
import matplotlib.pyplot as plt   
import numpy as np                  
```
![image](https://user-images.githubusercontent.com/59284238/133887891-b1c6a08c-58e7-4d0e-99b8-84716d7199f6.png)

## To train the naïve Bayes classifier
step 0) - Collect tweet samples / corpus <br>
step 1) Get or annotate a dataset with positive and negative tweets <br>
Step 2) Preprocess the tweets: process_tweet(tweet)= 
Lowercase
Remove punctuation, urls, names
Remove stop words
Stemming
Tokenize sentences <br>
Step 3) Compute freq(w, class) <br>
step 4) Get P(w|pos), P(w|neg) <br>
step 5) Get λ(w) = log (P(w|pos) / P(w|neg)) <br>
step 6) Compute logprior - This ratio between positive and negative tweets is called the prior ratio.
These ratios are key for Naive Bayes <br>
Positive words have a ratio larger than 1 <br>
Negative words have a ratio lower than 1 <br>
Neutral words have a ratio of 1 <br>

## Applications of Naive Bayes
1) Author identification
2) Spam filtering 
3) Information retrieval 
4) Word disambiguation etc.

# Biblography
I would like to thank Andrew Ng for giving guidance on the course and deeplearning.ai on coursera 
https://www.coursera.org/account/accomplishments/certificate/X6L7L32PXGPC

