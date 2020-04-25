#Restaurant Review Model using NLP : NLTK
#Here we create an ML model to predict for any review if it is '+ve' or '-ve' 


#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing dataset which contains the reviews of restaurant
#in file -> review 3 spaces then 1->postive review 0->negative review
#here tsv file used inplace of csv coz reviews might have , which is the delimiter 
#whereas in tsv delimeter is tab space
#quoting=3 here ignores the double codes
data=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3)



#Step 1 -> cleaning the text 
#cleaning the text -> must step in NLP Algo.

#lib -> clean txt efficiently
import re
import nltk
#for 1.
nltk.download('stopwords')    
from nltk.corpus import stopwords
#for 2.
from nltk.stem.porter import PorterStemmer

#corpus -> collection of words -> here reviews after cleaning it
courpus=[]

#taking all the review and cleaning it
for i in range(0,1000):
    #review is updated one with clean words
    # re.sub() ->remove unwanted items like number, symbols etc.. and contains only letters
    #parameters ,1: include oly letters ,2: replace taken items by space ,3:sentance to be cleaned.
    review=re.sub('[^a-zA-Z]',' ',data['Review'][i])
    #making into lowercase
    review=review.lower()
    #1.removing non significant words(all articals n prepositions) and stemming
    #2.stemming -> taking the root word for all words in the review
    #split the review into different words
    review=review.split() 
    ps=PorterStemmer() 
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #making as string again ' '. means separaeted by space
    review=' '.join(review)
    courpus.append(review)
    


#Step 2 -> Bag of words which containts only relevant words -> tokenization
#bag of words -> taking all unique words and creating sparse matrix
    #sparse matrix -> each word in a coloumn and each review in row 
        #if word is found v fill it with 1 else 0

#the words and review relation -> classification model
#Independnt variable -> each word in review dataset
#dependnt variable -> result wheather review is postive r negative

#bag of words model i.e tokenization
from sklearn.feature_extraction.text import CountVectorizer
#cv parameter max_features -> keep most frequent words 
cv=CountVectorizer(max_features=1500)

#X -> Independent Variable
#X -> sparse matrix fitiing into the cv model , .toarray() -> for matrix
X=cv.fit_transform(courpus).toarray()

#Y -> Depentent Variable i.e result 0 or 1
Y=data.iloc[:,1].values


#Step 3 -> creating our ML model to train it
#Classification model -> GaussenNB model v r using here

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# parameter -> 1.Ind var 2.Dep var 3.20% ratio test:train 4.for same answer
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size = 0.20, random_state = 0)

#Naive bayes Model (Classification)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()

#Fitting the training set to the GNB classifier
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred=classifier.predict(X_test)

# Making the Confusion Matrix (cm) -> for checking predicted result and original result
    # correct prediction=[0,0]+[1,1]=73
    # incorrect prediction=[0,1]+[1,0]-27
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, Y_pred)


################################################################################################################
###########################USER CODE############################################################################
################################################################################################################


def predict_review(review):
    review=re.sub('[^a-zA-Z]',' ',review)
    review=review.lower()
    review=review.split() 
    #ps -> PorterStemmer object
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    #cv -> instance of CountVectorizer
    review=cv.transform([review]).toarray()
    result=classifier.predict(review)
    return result

user_review=input("Enter Review to be predicted: ")
result=predict_review(user_review)
print(result)
if result == 0:
    print("Review is Negative")
else:
    print("Review is Postive")