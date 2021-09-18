# Twitter-Sentiment-Analysis-Project by using NLP 
Hi!
I am Saksham Garg, currently studying in 3rd year and persuing my Bachelor's of technology in Information Technology

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
from utils import process_tweet, build_freqs 
```
![image](https://user-images.githubusercontent.com/59284238/133887891-b1c6a08c-58e7-4d0e-99b8-84716d7199f6.png)


