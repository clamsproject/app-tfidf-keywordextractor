from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tfidf import sort_coo, extract_topn_from_vector, get_keywords

with open("/Users/selenasong/Desktop/CLAMS/tfidf-kw-detection/200304.newshour-transcript-casey/cpb-aacip_507-9p2w37mh8d.txt", 'r') as f:
    text = f.read()
    # text_list = []
    # text_list.append(text)

with open("/tfidf-kw-detection/app-tfidf-keyword-detector/tfidf_vectors.pkl", 'rb') as idf_file:
    idf_values = pickle.load(idf_file)

with open("/tfidf-kw-detection/app-tfidf-keyword-detector/features.pkl", 'rb') as feature_file:
    feature_names = pickle.load(feature_file)

cv = CountVectorizer(vocabulary=feature_names)
tfidf_transformer = TfidfTransformer()
tfidf_transformer.idf_ = idf_values


keywords = get_keywords(text, feature_names, 10, tfidf_transformer, cv)

print(keywords)
