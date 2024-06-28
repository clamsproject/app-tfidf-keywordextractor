import os
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle as pkl

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
    for txt_file in os.listdir(directory):
        txt_file_path = os.path.join(directory, txt_file)
        if txt_file_path.endswith('.txt'):
            with open(txt_file_path, 'r') as f:
                lines = f.read().strip()
                all_text_list.append(lines)

    return all_text_list

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", action='store', help="path to the directory of all text files for generating idf scores ")
    parser.add_argument("--idfFeatureFile", action='store', default='idf_feature_file.pkl', help='file name of the idf vectors. If nothing is passed in, the file in the current directory named as "idf_feature_file.pkl" is used')
    parser.add_argument("--maxDf", action='store', type=float, default=0.85, help='maximum document frequency. Default value is 0.85. Max value is 1.0')
    parsed_args = parser.parse_args()
    docs = lists_generator(parsed_args.dataPath)
    cv = CountVectorizer(max_df=parsed_args.maxDf, stop_words='english')
    word_count_vector = cv.fit_transform(docs)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    feature_names = cv.get_feature_names_out()
    idf_feature_values = {"idf_values": tfidf_transformer.idf_, "feature_dict": feature_names}

    with open(parsed_args.idfFeatureFile, 'wb') as f:
        pkl.dump(idf_feature_values, f)

