# Project 2: Disaster Response Pipelines
Project Submission : Data Science at Udacity.


## Installations.
This project was written on the [Project Workspace IDE](https://classroom.udacity.com/) provided and Jupyter application by python version 3.9.5. To check python by `python --version`. 
Using the libary `pandas`, `sqlite3` to loads the data. `nltk`, `pandas` to clean the data. `sklearn` to analyze the data. `pickle`, `pandas` to save the data.
`flask`, `json`, `plotly` to visualyzation. You can install the library as a terminal on the iOS system as follows:
```
pip install nltk
pip install jsons
pip install pysqlite3
pip install pickle5
pip install plotly
pip install flask
pip install sklearn
pip install sqlalchemy
```


## Project Details.
This project using disaster data from [Appen](https://appen.com/) to analyze by ETL piepline, Machine Learning pipeline and Flask visualzation, respectively.

__1. ETL Pipeline Prepareation.__ 

In a Python script, `process_data.py` or `ETL Pipeline Preparation.ipynb`, write a data cleaning pipeline that: 
* Loads the `messages` and `categories` datas.
* Marges datas.
* Cleaning the data.
* Save to SQLite database.

__2. Machine Learning Pipeline Prepareation.__

In a Python script, `train_classifier.py` or `ML Pipeline Preparation.ipynb`, write a machine learning pipeline by sklearn that:
* Loads data from SQLite database to result file of `process_data.py`.
* Splits the data to test and train.
* Cleaning the data.
* Train and test a model using by GridSearchCV
* Evaluate model.
* Save the data using pickle `.pkl`.


__3. Flask App__

In a Python script, `run.py`, write a flask web application that:
* Modify file paths for database `classifier.pkl` to result of `train_classifier.py`
* Add visualizations using Plotly in the web app.



## File Description.
The coding for this project can be completed using the Project Workspace IDE provided. Here's the file structure of the project:
```
- app
| - template
| |- master.html
| |- go.html
|- run.py

- data
| - disaster_categories.csv           # data to process_data.py
| - disaster_messages.csv             # data to process_data.py
| - process_data.py
| - DisasterResponse.db               # database result from process_data.py and using in the run.py
| - categories.csv                    # data to train_classifier.py
| - messages.csv                      # data to train_classifier.py

- models
| - train_classifier.py
| - classifier.pkl                    # saved model.

- ETL Pipeline Preparation.ipynb
- ML Pipeline Preparation.ipynb
- README.md
```

To run the project on [http://0.0.0.0:3000/](https://052b8c06e71c40e99045311a7e35ff96-3000.udacity-student-workspaces.com/) by
 
STEP 1: Run __process_data.py__, Python scripts can be able to run with arguments as,
* `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`

STEP 2: Run __train_classifier.py__ ,Python scripts can be able to run with arguments as ,
* `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

STEP 3: Run __run.py__, Python scripts can be able to run with arguments as,
* `python run.py`



## How to Interract with Your Project.
I learned and understood the disaster data to analyze that:
* ETL: Load data, cleaning data and save to SQLite databases.
* ML: Load data form  SQLite database, analyze by using classifier model and save to pickle.
* Web app: Get the data form pickle to visualization on web application.



## Copyright and license.
Coldding learn by [Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
and using code at [Scikit-learn](https://scikit-learn.org/stable/), 
[NLTK](https://www.nltk.org/), 
[Flask](https://flask.palletsprojects.com/en/2.0.x/)




