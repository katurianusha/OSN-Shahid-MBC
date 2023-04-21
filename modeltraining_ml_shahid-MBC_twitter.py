import pandas as pd
import numpy as np
import csv
import io

from IPython.display import FileLink
from google.cloud import storage

# Setting credentials using the downloaded JSON file
client = storage.Client.from_service_account_json(json_credentials_path='/home/anusha_k2/service-account.json')

# Replace with your Google Cloud Storage bucket and file path
bucket_name = "metrics-scraping-bucket"
file_path = "Airtel-reviews train.csv"
file_path1 = "Twitter_Shahid-MBC_Reviews.csv"
# Initialize a client object
client = storage.Client()

# Retrieve the bucket containing the file
bucket = client.bucket(bucket_name)

# Retrieve the blob representing the file
blob = bucket.blob(file_path)
blob1 = bucket.blob(file_path1)

# Read the contents of the file into a Pandas DataFrame
filecontent = blob.download_as_string()
filecontent1 = blob1.download_as_string()
df = pd.read_csv(io.BytesIO(filecontent))
df1 = pd.read_csv(io.BytesIO(filecontent1))
# for removing the empty rows from a dataframe df1 
df1 = df1.dropna()
df2 = pd.concat([df, df1], axis=0)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from IPython.display import FileLink

train_labels = df2['Label'][:301]
test_data = df2['Review'][301:]
test_labels = df2['Label'][301:]
train_data = df2['Review'][:301]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, train_labels)
predicted_labels_test = nb_classifier.predict(X_test)
predicted_labels_train = nb_classifier.predict(X_train)
predicted_labels_test = np.array(predicted_labels_test)

all_labels = np.concatenate((predicted_labels_train, predicted_labels_test))
df3 = pd.DataFrame({'Review': test_data, 'Predicted Label': predicted_labels_test})
df3.to_csv('Twitter_shahid-MBC_PredictedFinal.csv', index=True)
