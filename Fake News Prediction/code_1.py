import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt  # For Evaluting the trained Model  
from sklearn.feature_extraction.text import TfidfVectorizer # To Implement TF-IDF
import joblib #loding and sving the model
from sklearn.svm import LinearSVC
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

t_news = pd.read_csv('True.csv')
f_news = pd.read_csv('Fake.csv')

t_news['label'] = 1
f_news['label'] = 0

#Preprocessing the data 
data_merge = pd.concat([t_news, f_news], axis = 0)

# Display the first and last two rows
data_merge.head(2)
data_merge.tail(2)

# Display column names
data_merge.columns

Data =  data_merge.drop(['title','subject','date'],axis = 1)

data_merge.head()
data_merge.tail()

# Shuffling the dataset
shuffled_data = Data.sample(frac=1).reset_index(drop=True)
shuffled_data

new_df = shuffled_data.copy()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

 
def transform(text):
    text = text.lower()  # Convert text to lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    # Lemmatize, remove stopwords and punctuation
    tokens = [lemmatizer.lemmatize(word, "v") for word in tokens if word not in stop_words and word not in punctuations]
    return " ".join(tokens)

new_df["text"] = new_df["text"].apply(transform)

new_df.head(2)

# Split data into training + validation and test sets
X_temp, X_test, y_temp, y_test = train_test_split(new_df['text'], new_df['label'], test_size=0.2, random_state=3)

# Split training + validation set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=3)


# TF-IDF Represent how important a word is in all document
tfidf = TfidfVectorizer()
#build a vocabulary for training data 
tfidf.fit(X_train)
#tfidf.vocabulary_   #TO VIEW THE VOCABULARY

# tranforming X_train, X_val & X_test into numerical vectores
X_train_tfidf = tfidf.transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)


#Model Supper Vector Machine SVM
Model = LinearSVC(C=1.0, max_iter=1000, tol=1e-4)
Model.fit(X_train_tfidf, y_train)


       # the following code is used to evalute the dataset using validate dataset
#y_val_pred = Model.predict(X_val_tfidf)

                       # EValuting the metrics 
#accuracy = accuracy_score(y_val, y_val_pred)
#precision = precision_score(y_val, y_val_pred)
#recall = recall_score(y_val, y_val_pred)
#f1 = f1_score(y_val, y_val_pred)

                # Predict probabilities for the positive class
#y_test_pred_proba = Model.decision_function(X_test_tfidf)

                #  Predict the class labels based on a default threshold (0.5)
#y_test_pred = Model.predict(X_test_tfidf)

                #  Generate the confusion matrix
#cm = confusion_matrix(y_test, y_test_pred)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])
#disp.plot(cmap=plt.cm.Blues)
#plt.show()

               # ROC Curve and Calculate AUC
#fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
#auc_score = roc_auc_score(y_test, y_test_pred_proba)
#plt.figure()
#plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
#plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC)')
#plt.legend(loc="lower right")
#plt.show()

# Save the trained model to a file
joblib.dump(Model, 'SVC_model.pkl')

# Load the model from the file
Model = joblib.load('SVC_model.pkl')

# Save the vectorizer to a file
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Load the vectorizer from the file
tfidf = joblib.load('tfidf_vectorizer.pkl')