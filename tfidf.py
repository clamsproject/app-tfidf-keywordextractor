import os
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle as pkl
import pandas as pd

# data_path = '/Users/selenasong/Desktop/CLAMS/tfidf-kw-detection/200304.newshour-transcript-casey'
# tfidf_pkl_path = '/tfidf-kw-detection/app-tfidf-keyword-detector/tfidf_vectors.pkl'
# features_path = '/tfidf-kw-detection/app-tfidf-keyword-detector/features.pkl'

def find_problematic_files(directory):
    """
    Find all files that are either mmif style annotation or whose status is '404 not found'
    These files are in JSON style, so this function is to record any file that contains '{'
    Helper function for delete_problematic_files.
    """
    problematic_files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                text = f.read()
                if "{" in text:
                    problematic_files.append(file_path)

    return problematic_files

def delete_problematic_files(directory):
    """
    Delete all the problematic files
    """
    list_of_problematic_files = find_problematic_files(directory)
    for file in list_of_problematic_files:
        os.remove(file)

    # Double check whether there is other problematic files
    return find_problematic_files(directory)

def lists_generator(directory):
    """
    Put all texts into one list
    """
    all_text_list = []
    all_files_list = os.listdir(directory)
    for txt_file in all_files_list:
        txt_file_path = os.path.join(directory, txt_file)
        if txt_file_path.endswith('.txt'):
            with open(txt_file_path, 'r') as f:
                lines = f.read().strip()
                all_text_list.append(lines)

    return all_text_list, all_files_list

# docs, files = lists_generator(data_path)

# Creating the IDF
# CountVectorizer to create a vocabulary and generate word counts
#create a vocabulary of words,
#ignore words that appear in 85% of documents,
#eliminate stop words

# cv = CountVectorizer(max_df=0.85, stop_words='english')
# word_count_vector = cv.fit_transform(docs)

# TfidfTransformer to Compute Inverse Document Frequency (IDF)
# tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
# tfidf_transformer.fit(word_count_vector)
# feature_names = cv.get_feature_names_out()

# with open(tfidf_pkl_path, 'wb') as f:
#     pkl.dump(tfidf_transformer.idf_, f)
#
# with open(features_path, 'wb') as file:
#     pkl.dump(feature_names, file)

# Computing Tf-IDF and Extracting Keywords
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

def get_keywords(doc, feature_names, topn, tfidf_transformer, cv):
    # generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, topn=topn)

    return keywords

def print_results(idx,keywords):
    # now print the results
    print("\n=====Text=====")
    print(docs[idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k, keywords[k])
#
# idx=0
# keywords = get_keywords(docs[idx], feature_names, 10, tfidf_transformer)
# print_results(idx,keywords)


# Generate keywords for a batch of documents
#generate tf-idf for all documents in your list. docs_test has 500 documents
# tf_idf_vector = tfidf_transformer.transform(cv.transform(docs))
#
# results = []
# for i in range(tf_idf_vector.shape[0]):
#     # get vector for a single document
#     curr_vector = tf_idf_vector[i]
#
#     # sort the tf-idf vector by descending order of scores
#     sorted_items = sort_coo(curr_vector.tocoo())
#
#     # extract only the top n; n here is 10
#     keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
#
#     results.append(keywords)
#
# df = pd.DataFrame(zip(docs, results),columns=['doc', 'keywords'])
# print(df[:1000])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", action='store', help="path to the directory of all text files for generating idf scores ")
    parser.add_argument("--output_idf_file_name", action='store', help="file name of the idf vectors")
    parser.add_argument("--output_feature_file_name", action='store', help="file name of the features")
    parser.add_argument("--max_df", action='store', type=float, default=0.85, help="maximum document frequency")
    parsed_args = parser.parse_args()
    docs, files = lists_generator(parsed_args.data_path)
    cv = CountVectorizer(max_df=parsed_args.max_df, stop_words='english')
    word_count_vector = cv.fit_transform(docs)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    feature_names = cv.get_feature_names_out()

    with open(parsed_args.output_idf_file_name, 'wb') as f:
        pkl.dump(tfidf_transformer.idf_, f)

    with open(parsed_args.output_feature_file_name, 'wb') as file:
        pkl.dump(feature_names, file)

