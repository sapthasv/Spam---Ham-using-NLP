# importing the library
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd


# importing the data set
messages1 = pd.read_csv('spam.csv',encoding='latin-1')
messages =messages1.iloc[:,[0,1]]

# changing cols with rename() 
messages = messages.rename(columns = {"v1": "Result", 
                                      "v2":"Message"}) 


# cleaning the text 
ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []

#sing stemming
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]',' ' , messages['Message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    review = ' '.join(review) 
    corpus.append(review)

'''
# using lematization
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]',' ' , messages['Message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))] # here we can use lematization or stemming
    review = ' '.join(review) 
    corpus.append(review)
'''



# creating the TF-IDF Model
from sklearn.feature_extraction.text import TfidfVectorizer
TF_IDF = TfidfVectorizer()   # in tf-idf seperate craetes the seperate precision to the words i.e 0.5,0.75,0.3612
X = TF_IDF.fit_transform(corpus).toarray()



'''
# creating the BOW model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=6000)     # in BOW model it creates the same priority to all the words i.e 0 or 1
X = cv.fit_transform(corpus).toarray()
'''

y = pd.get_dummies(messages['Result'])
y= y.iloc[:,0].values

# for visulization  and exporting to excel
visulize =pd.DataFrame(X,y)
export_excel = visulize.to_excel (r'C:\Users\dell\Desktop\NLP_export_dataframe.xlsx', index = None, header=True) # Dont forget to add .xlsx at the end of the path


# train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state=0)

'''
# fitting the model
# for NLP naive bayes is the most used algorithm
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB # here accuracy is 88%
classifier = GaussianNB()
classifier.fit(X_train, y_train)
'''

# fitting the multinomial naive bayes
from sklearn.naive_bayes import MultinomialNB # this multinomial naive bayes accuracy is more comapred to gaussian i.e 96%
classifier = MultinomialNB()
classifier.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#checking the acuracy score
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred) * 100
