# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:21:32 2018

@author: CAIA102
"""

import tweepy
import pandas as pd
from textblob import TextBlob

## Twitter credentials
consumer_key = "TdOunCxvfCMDaNPE6qLsmMCNI"
consumer_secret = "RquDGTgxhV97NFWv2PtuKCz0vPir3SdEhsbShcToeQRshfhruK"
access_token = "2894385181-OWXRLnKEX4njDWBVw4sRLfC3GBXHeCu1qm7Leky"
access_token_secret = "RlP59jIvbSQxfIvw65fq0fFz0C8g6dfLhKuEmzhNI2f35"

#Set up an instance of Tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#Fetching the tweets and appending them into a list
twt=[]
for tweet in tweepy.Cursor(api.search,q="ipad Pro",count=100,
                           lang="en",exclude_replies=True
                           ).items(1000):
    print (tweet.created_at, tweet.text)
    twt.append(([tweet.created_at, tweet.text.encode('ascii','ignore')]))




#performing sentimental analysis
import re 

sent_lst=[]        
for i in range(0,len(twt)):     
    sent_lst.append(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)','',str(twt[i][1]))+'.')
     
   
c =[]
comments = []
score=[]

for i in sent_lst:
    print(i.replace('bRT',''))     
    review = TextBlob(i.replace('bRT',''))
    score.append(review.sentiment.polarity) #Gives the Sentiment Score for each tweet
    if review.sentiment.polarity >0:
         comments.append('Positive')
    elif review.sentiment.polarity == 0:
         comments.append('Neutral')
    else:
         comments.append('Negative')
len(comments)      

#Count Plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(comments,order=['Positive','Neutral','Negative'],palette=['Green','Blue','Red'])
#plt.figure(figsize=(10,10))
plt.title("")
plt.show() 
comments.count("Positive")

# Pie Plot
labels = 'Positive', 'Neutral', 'Negative'
sizes = [comments.count("Positive"), comments.count("Neutral"),comments.count("Negative")]
colors = ['Cyan','yellow','Red']
explode = (0.1, 0, 0)  # explode 1st slice
plt.figure(figsize=(10,10))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%2.1f%%', shadow=True, startangle=150)
plt.axis('equal')
plt.title("")
plt.show()





################################Topic Modelling#################################


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
stop1=list(stop)
stop1.append('Ipad')
stop1.append('ipad')
stop1.append('pro')
stop1.append('Apple')
stop1.append('apple')
stop1.append('Pro')
stop1.append('mpdy2lla')
stop1.append('using')
stop1.append('ut')
stop1.append('win')
stop1.append('Ipad')
stop1.append('rt')
#stop1.append('video')
stop1.append('iphone')
stop1.append('macbook')
stop1.append('octoer')
stop1.append('im')
stop1.append('mac')
stop1.append('news')
stop1.append('enter')
stop1.append('Rt')
stop1.append('2018')
stop1.append('another')
stop1.append('e')
stop1.append('io')
stop1.append('aout')
stop1.append('free')
stop1.append('see')
stop1.append('liked')
stop1.append('start')
stop1.append('know')
stop1.append('day')
stop1.append('check')

#stop1.append('microsoft')
exclude = set(string.punctuation)

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("bRT", " ", tweet).split())


clnt=[]
clnf=[]
for i in sent_lst:
    #print(clean_tweet(i))
    clnt.append(clean_tweet(i))
for i in clnt:
    clnf.append(' '.join(re.sub("b", "", i).split()))

lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop1])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in clnf]  
        
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=10, num_words=5))

topic_modelling=pd.DataFrame(ldamodel.print_topics(num_topics=10, num_words=5))

tp_plot=ldamodel.print_topics(num_topics=10, num_words=5)


t1=tp_plot[0][1]
t2=tp_plot[1][1]
t3=tp_plot[2][1]
t4=tp_plot[3][1]
t5=tp_plot[4][1]
t6=tp_plot[5][1]
t7=tp_plot[6][1]
t8=tp_plot[7][1]
t9=tp_plot[8][1]
t10=tp_plot[9][1]

def cap(a):
    num=[]
    txt=[]
    for i in range(0,a.find('*')):
        #print(a[i])
        num.append(a[i])
    for i in range(a.find('"'),a.find(' ')):
        #print(a[i])
        txt.append(a[i])
    return float("".join(num))*100,"".join(txt)


y=[cap(t1)[0],cap(t2)[0],cap(t3)[0],cap(t4)[0],cap(t5)[0],cap(t6)[0],cap(t7)[0],cap(t8)[0],cap(t9)[0],cap(t10)[0]]
x=[cap(t1)[1],cap(t2)[1],cap(t3)[1],cap(t4)[1],cap(t5)[1],cap(t6)[1],cap(t7)[1],cap(t8)[1],cap(t9)[1],cap(t10)[1]]
width = 1/1.5
for i in range(0,len(x)):
    x[i]=x[i].replace('"','')
plt.bar(x, y, width, color="Green")
plt.title("Most used Topics ")
plt.show()

###############################Automatic Summarization####################################
import re
import nltk
#nltk.download()
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords



#The input for summarization must be one single paragraph of text.Hence, we join all the cleaned and tokenised tweets as one single paragraph using JOIN command. Each tweet is ended by a fullstop(.) and a new line character(/n)
c1=' '.join(clnf)
c2=summarize(twt)
TextBlob(c2).correct() #Spelling Correction
print ('Summary:')
print (c2)
print ('\n Keywords:')
print (keywords(str(clnf)))
keywords=keywords(str(clnf))

nltk.download()

# =============================================================================
# 
# ###########################Automatic Summarization for text ##################################
# from gensim.summarization.summarizer import summarize
# from gensim.summarization import keywords
#  
# import requests
# 
# # getting text document from file
# fname="C:\\Users\\CAIA102\\Desktop\\IPad.txt"
# with open(fname, 'r') as myfile:
#       text=myfile.read()
#       
#       
# # getting text document from Internet
# text = requests.get('https://www.techradar.com/reviews/new-ipad-2018-review').text
# 
# #getting text document from web, below function based from 3
# from bs4 import BeautifulSoup
# from urllib.request import urlopen
#  
# # =============================================================================
# # def get_only_text(url):
# #  """ 
# #   return the title and the text of the article
# #   at the specified url
# #  """
# #  page = urlopen(url)
# #  soup = BeautifulSoup(page, "lxml")
# #  text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
# #  return soup.title.text, text    
# #  
# #   
# # =============================================================================
#  
# url='https://www.techradar.com/reviews/new-ipad-2018-review'
# text = get_only_text(url)
#       
#       
#     
#       
#       
#       
#       
# print ('Summary:')   
# print (summarize(str(text), ratio=0.1))
# =============================================================================
# =============================================================================
#  
# print ('\nKeywords:')
#  
# # higher ratio => more keywords
# print (keywords(str(text), ratio=0.01)
# =============================================================================
