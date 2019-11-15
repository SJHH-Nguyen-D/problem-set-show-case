
import pandas as pd 
import spacy
import re

# Constants
questions_path = "../input/Questions.csv"
answers_path = "../input/Answers.csv"
features_include = ['Score', 'Title', 'Body']


def load_dataset_as_dataframe(path, n_obs, features_include):
    dataframe = pd.read_csv(path, 
                            nrows=n_obs, 
                            usecols=features_include,
                            encoding='latin1')
    dataframe = dataframe.dropna()
    return dataframe

def clean_text()
    pass

def all_process_text(dataframe):
    clean_text()
    pass

def generate_text_summary():
    pass

def main():
    df_questions = load_dataset_as_dataframe(questions_path, 1000, features_include)
    
    generate_text_summary()

if __name__ == "__main__":
    main()