import pandas as pd
import sklearn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import re
from xlwt import Workbook

#wb = Workbook()
#sheet1 = wb.add_sheet('Sheet 1')
#sheet1.write(0, 0, 'Neurons')
#sheet1.write(0, 1, 'Accuracy')
#sheet1.write(0, 2, 'Precision')
#sheet1.write(0, 3, 'Recall')
#sheet1.write(0, 4, 'F-Score')
#row = 1


data = pd.read_csv("/Users/varunnegandhi/Documents/Data Mining/IMDB Dataset.csv", encoding='latin1')
data['SentimentText'] = data['SentimentText'].str.lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['SentimentText'])
print("Fitting complete")

print(data.head())

for i in range(len(data)):
    data['SentimentText'][i] = re.sub('<.*?>', ' ', data['SentimentText'][i])
    data['SentimentText'][i] = tokenizer.texts_to_sequences([data['SentimentText'][i]])
    data['SentimentText'][i] = sum(data['SentimentText'][i], [])

data['Sentiment'] = data['Sentiment'].map({'positive': 1, 'negative': 0})

print(data.head())
print(type(data['SentimentText'][0]))

X = data['SentimentText']
y = data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)
print(len(X_train))

top_words = 150000
max_words = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

#for hn in range(150, 551, 10):
model = Sequential()
model.add(Embedding(top_words, 30, input_length=max_words))
model.add(Flatten())
    #model.add(Dense(1024, input_dim=len(data.columns)-1, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(hn, activation='relu'))
    #model.add(Dense(hn, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.metrics_names)
model.fit(X_train, y_train, epochs=2, batch_size=150, validation_split=0.2)
    #print(model.summary())

    # evaluate keras model
y_pred = model.predict(X_test)
    #print(y_pred.shape)
    # print("Length categories:", len(y_pred))
y_pred_labels = np.ndarray.argmax(y_pred, axis=1)
comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_labels})
print("Accuracy:", round(accuracy_score(comp['Actual'], comp['Predicted'])*100, 2))

score = sklearn.metrics.classification_report(comp['Actual'], comp['Predicted'], output_dict=True)


    #sheet1.write(row, 0, hn)
    #sheet1.write(row, 1, round(accuracy_score(comp['Actual'], comp['Predicted']) * 100, 2))
    #sheet1.write(row, 2, round(score['weighted avg']['precision'] * 100, 2))
    #sheet1.write(row, 3, round(score['weighted avg']['recall'] * 100, 2))
    #sheet1.write(row, 4, round(score['weighted avg']['f1-score'] * 100, 2))
    #row = row + 1

data_test = pd.read_csv("/Users/varunnegandhi/Documents/Data Mining/dataset.csv", encoding='latin1')
data_test['SentimentText'] = data_test['SentimentText'].str.lower()

for i in range(len(data_test)):
    data_test['SentimentText'][i] = re.sub('<.*?>', ' ', data_test['SentimentText'][i])
    data_test['SentimentText'][i] = tokenizer.texts_to_sequences([data_test['SentimentText'][i]])
    data_test['SentimentText'][i] = sum(data_test['SentimentText'][i], [])

X_data_test = data_test['SentimentText']
y_data_test = data_test['Sentiment']

X_data_test = sequence.pad_sequences(X_data_test, maxlen=max_words)
print(data_test.head())

y_pred = model.predict(X_data_test)
print(y_pred.shape)
print("Length categories:", len(y_pred))
y_pred_labels = np.ndarray.argmax(y_pred, axis=1)
comp = pd.DataFrame({'Actual': y_data_test, 'Predicted': y_pred_labels})
print("Accuracy:", round(accuracy_score(comp['Actual'], comp['Predicted'])*100, 2))
print(y_pred)
score = sklearn.metrics.classification_report(comp['Actual'], comp['Predicted'], output_dict=True)
    #sheet1.write(row, 0, hn)
    #sheet1.write(row, 1, round(accuracy_score(comp['Actual'], comp['Predicted']) * 100, 2))
    #sheet1.write(row, 2, round(score['weighted avg']['precision'] * 100, 2))
    #sheet1.write(row, 3, round(score['weighted avg']['recall'] * 100, 2))
    #sheet1.write(row, 4, round(score['weighted avg']['f1-score'] * 100, 2))
    #row = row + 1
    #wb.save('DM-Project-Values3.xls')
