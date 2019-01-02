import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function is to load two datasets from the file and merge it.
    
    arg:str  two datasets path
    return: dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    '''
    The function is to wrangle the dataset and drop the duplicates.
    arg: dataframe
    return: dataframe
    '''
    categories = df['categories'].str.split(';', expand = True) # split categories in to separate category columns
    row = categories.loc[0, :] #select the first row of the dataframe
    category_colnames = [x[:-2] for x in row] # extrat the column names
    categories.columns = category_colnames # rename the column names
    
    # conver category values to numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str.extract('(\d)', expand = False)
        categories[column] = pd.to_numeric(categories[column])
        
    df = df.drop('categories', axis = 1) # drop the original categories column from df
    df = pd.concat([df, categories], axis = 1)
    
    df = df.drop_duplicates() # drop the duplicates
    return df


def save_data(df, database_filename):
    
    '''
    The function is to save the dataframe to sqlite database.
    args: dataframe, database file name
    
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename, engine, index = False)
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()