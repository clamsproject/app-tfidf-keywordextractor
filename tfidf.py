import os
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle as pkl
from clams_utils.aapb.newshour_transcript_cleanup import file_cleaner

def read_newshour_transcript(newshour_transcripts_directory):
    """
    Given a directory of NewsHour transcripts,
    return a list that contains all the cleaned transcripts for later model training
    """
    all_text_list = []
    for txt_file in os.listdir(newshour_transcripts_directory):
        txt_file_path = os.path.join(newshour_transcripts_directory, txt_file)
        if txt_file_path.endswith('.txt'):
            cleaned_lines = file_cleaner(txt_file_path)
            if cleaned_lines != None:
                all_text_list.append(cleaned_lines)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", action='store',
                        help='path to the directory of all transcripts used for generating idf scores.')
    # TODO: comment this argument back after the modelName parameter in metadata.py is solved
    # parser.add_argument("--idfFeatureFile", action='store', default='idf_feature_file.pkl',
    #                     help='file name of the idf vectors. If nothing is passed in, the file in the current directory '
    #                          'named as "idf_feature_file.pkl" is used')
    parser.add_argument("--maxDf", action='store', type=float, default=0.85,
                        help='maximum document frequency. Default value is 0.85. Max value is 1.0')
    parsed_args = parser.parse_args()
    docs = read_newshour_transcript(parsed_args.dataPath)
    cv = CountVectorizer(max_df=parsed_args.maxDf, stop_words='english')
    word_count_vector = cv.fit_transform(docs)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    feature_names = cv.get_feature_names_out()
    idf_feature_values = {"idf_values": tfidf_transformer.idf_, "feature_dict": feature_names}

    # TODO: change the pickle file to read after the argument is commented back
    with open('./idf_feature_file.pkl', 'wb') as f:
        pkl.dump(idf_feature_values, f)

