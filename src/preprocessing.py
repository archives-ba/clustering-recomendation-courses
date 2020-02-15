import pandas as pd
import numpy as np

def remove_nan_rows(data_frame):
    """
    Remove all rows with NaN values for any column

    Parameters
    ----------
    data_frame : Pandas.DataFrame
        Dataset to drop rows
    """
    return data_frame.dropna(how='any')

def remove_punctuation(data_frame, columns):
    """
    Remove all punctuation from the specified columns

    Parameters
    ----------
    data_frame : Pandas.DataFrame
        Data frame containing columns
    columns : Array.String
        Columns to remove punctuation
    """
    punctuation = ["{", "}", "(", ")", "[", "]", ".", ",", ":", ";", "...", "!", "?"]
    data_frame[columns] = data_frame[columns].replace(to_replace=punctuation, value=" ")
    return data_frame

def remove_stopwords(data_frame, columns):
    """
    Remove all stopwords from the specified columns

    Parameters
    ----------
    data_frame : Pandas.DataFrame
        Data frame containing columns
    columns : Array.String
        Columns to remove stopwords
    """
    stopwords = np.array(pd.read_json("../resource/stopwords-en.json")["stopwords"])
    for column in columns:
        data_frame[column] = data_frame[column].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stopwords]))
    return data_frame

def remove_abreviations(data_frame, columns):
    """
    Remove all stopwords from the specified columns

    Parameters
    ----------
    data_frame : Pandas.DataFrame
        Data frame containing columns
    columns : Array.String
        Columns to remove stopwords
    """
    data_frame[columns] = data_frame[columns].replace({"'ll": " "}, regex=True)
    return data_frame
    
def remove_hyphen(data_frame, columns):
    """
    Remove hyphens from the specified columns

    Parameters
    ----------
    data_frame : Pandas.DataFrame
        Data frame containing columns
    columns : Array.String
        Columns to remove hyphens
    """
    data_frame[columns] = data_frame[columns].replace({"-": " "}, regex=True)
    return data_frame

def remove_all_characters_except_numbers_alphabets(data_frame, columns):
    """
    Remove all characters except numbers & alphabets

    Parameters
    ----------
    data_frame : Pandas.DataFrame
        Data frame containing columns
    columns : Array.String
        Columns to remove characters
    """
    data_frame[columns] = data_frame[columns].replace({"[^A-Za-z0-9 ]+": ""}, regex=True)
    return data_frame


def combine_columns_fields(data_frame, columns):
    """
    Combine columns with space separator

    Parameters
    ----------
    data_frame : Pandas.DataFrame
        Data frame containing columns
    columns : Array.String
        Columns to combine
    """
    data_frame['combined_columns'] = data_frame[columns].agg(' '.join, axis=1)
    return data_frame