# Table of Contents
1. Installation
2. Project Motivation
3. Instructions
4. File Descriptions
5. Results
6. Licensing, Authors, and Acknowledgements

# Installation
python 3 .*  
pandas 0.20.3  
numpy 1.12.1  
nltk 3.2.5  
scikit-learn 0.19.1   
plotly 2.0.15  
Flask 0.12.2  

# Project Motivation
This project is about natural language processing(NLP) and classification. The data downloaded 
from [figure eight](https://www.figure-eight.com/data-for-everyone), containing messages from social medium and 36 categories.
In this project, I'am trying to use nltk package to process the messages and RandomForestClassifier of scikit-learn package to make 
classification. The results were displayed in web, so you can input the message and find its categories.
I hope this project can help rescue team make right actions when disasters come.

# Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# File Descriptions
This project contains three folders: app, data, models.  
app folder: this folder including templates folder which is template of htmls and a python file to build web;  
data folder: this folder including two csv files and a python file which is to process the data;  
models folder: this folder including a python file to build a classification model and save it.  


# Results
The results was displayed in web htmls


# Licensing, Authors, and Acknowledgements
[This project is under the Creative Commons Attribution-NonCommerical 3.0 Unported License](https://creativecommons.org/licenses/by-nc/3.0/)  
Auther: Wang Huan
