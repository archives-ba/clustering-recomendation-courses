import pandas as pd
import preprocessing
import pickle
from copy import deepcopy

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

def read_dataset():
    print("getting course_df")
    course_df = pd.read_csv("../resource/courses.csv")
    print("removing NaN rows")
    course_df = preprocessing.remove_nan_rows(course_df)
    print("removing hyphen")
    course_df = preprocessing.remove_hyphen(course_df, columns=['CourseId'])
    print("removing 'll")
    course_df = preprocessing.remove_abreviations(course_df, columns=['CourseId', 'Description', 'CourseTitle'])
    print("removing all characters except numbers alphabets")
    course_df = preprocessing.remove_all_characters_except_numbers_alphabets(course_df, columns=['CourseId', 'Description', 'CourseTitle'])
    print("combine columns")
    course_df = preprocessing.combine_columns_fields(course_df, columns=['CourseId', 'Description', 'CourseTitle'])
    return course_df

def train(course_df):
    print("tf-idf")
    tfidf = TfidfVectorizer(stop_words='english')
    x = tfidf.fit_transform(course_df['combined_columns'])
    pickle.dump(tfidf, open('tfidf_model.sav', 'wb'))
    print("k-means")
    centroid_number = 30
    clustering = KMeans(n_clusters=centroid_number, init='k-means++', max_iter=500, n_init=15)
    clustering.fit(x)
    print("saving")
    pickle.dump(clustering, open('clustering_model.sav', 'wb'))

def load_models():
    clustering = pickle.load(open('clustering_model.sav', 'rb'))
    tfidf = pickle.load(open('tfidf_model.sav', 'rb'))
    return tfidf, clustering

def cluster_predict(str_input, tfidf_model, clustering_model):
    x = tfidf_model.transform(list(str_input))
    prediction = clustering_model.predict(x)
    return prediction

def predict_all(course_df, tfidf_model, clustering_model):
    course_df['ClusterPrediction'] = ""
    course_df['ClusterPrediction'] = course_df.apply(lambda x: cluster_predict(course_df['combined_columns'], tfidf_model, clustering_model), axis=0)

def recommend_util(course_df, str_input, tfidf_model, clustering_model):
    str_input_formatted = str(str_input).replace("-", " ")
    print(str_input_formatted)
    temp_df = course_df[course_df['CourseId'] == str_input_formatted]
    str_input_formatted = list(temp_df['combined_columns'])
    prediction_inp = cluster_predict(str_input_formatted, tfidf_model, clustering_model)
    prediction_inp = int(prediction_inp)
    temp_df = course_df.loc[course_df['ClusterPrediction'] == prediction_inp]
    temp_df = temp_df.sample(10)
    return list(temp_df['CourseId'])

course_df = read_dataset()
# train(course_df)
tfidf_model, clustering_model = load_models()
predict_all(course_df, tfidf_model, clustering_model)
print(recommend_util(course_df, 'aspdotnet-advanced-topics', tfidf_model, clustering_model))