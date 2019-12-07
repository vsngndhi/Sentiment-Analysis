# Sentiment-Analysis
This is my project for Data Mining Course wherein I predict the sentiment from text reviews. It involves solving the problem using 2 approaches - Machine Learning SVM and Deep Learning - Keras

# Training Data
The source of training data is: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Testing Data
dataset.csv is the test data for this project

# DMProject2.ipynb:
This is the file which includes experimenting with SVM cost factor 0.1,1,10 and 100. Also it includes kernel type like linear, poly, rbf and sigmoid

# DMProject.ipynb
This includes experimenting with SVM with linear, rbf and sigmoid kernels with cost factor C values from 110 to 150 

# DM_Project.py
This is a python file dealing with deep learning keras. Here experiments were conducted by keeping embedding layer, embedding plus 1 hidden layer and embedding plus 2 hidden layers in the sequential keras model

# DM-Project-Values1.xls
This is output file containing values for accuracy, precision, recall and F-Score for embedding layer

# DM-Project-Values2.xls
This is output file containing values for accuracy, precision, recall and F-Score for embedding layer plus 1 hidden layer
To test this layer uncomment lines 56,62,82 and 111 in DM_Project.py file. The output is train data followed by test data in each line.

# DM-Project-Values3.xls
This is output file containing values for accuracy, precision, recall and F-Score for embedding layer plus 2 hidden layers
To test this layer uncomment lines 56,62, 63,82 and 111 in DM_Project.py file. The output is train data followed by test data in each line.

# DM-Presentation.pptx
Presentation for the entire project

# DM-Project-Report.docx
Complete report of the project
