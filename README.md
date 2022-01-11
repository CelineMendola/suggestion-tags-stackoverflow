# suggestion-tags-stackoverflow

This project aims to create a tag suggestion tool for posts in stackoverflow, using NLP techniques.

## Description

The data has been retrieved from stackexchange explorer. It is composed of posts written in stackoverflow since 2011.  We can find theses posts in QueryResults.csv in the data folder.  
The P5_01_notebookexploration.ipynb file aims to clean the text data, remove stopwords and to lemmatize text. Then, we make some analyses of the most frequent words in 
posts and tags.  
In P5_02_notebooktest.ipynb, we tested two types of models :  
- Topic modelisation with LDA and NMF  
- Multilabel classification  
The several models can be compared with the test.xlsx in the data folder. It is composed of predictions for about fifty posts unseen by the models. 

## Deployment whith Flask

In the flaskapp folder, the final model is implemented in model.py. It generates the preprocessing and prediction models in a pkl format. 
Running app.py gives a localhosted user interface, styled by the home.html file in the templates folder. 
The requirements.txt and Procfile files where created in order to deploy the application on the internet via heroku. 

