"""
DELETE THIS MODULE STRING AND REPLACE IT WITH A DESCRIPTION OF YOUR APP.

app.py Template

The app.py script does several things:
- import the necessary code
- create a subclass of ClamsApp that defines the metadata and provides a method to run the wrapped NLP tool
- provide a way to run the code as a RESTful Flask service 


"""

import argparse
import logging

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# For an NLP tool we need to import the LAPPS vocabulary items
from lapps.discriminators import Uri

# Import functions from tfidf.py
from tfidf import sort_coo, extract_topn_from_vector, get_keywords


class TfidfKeywordDetector(ClamsApp):

    def __init__(self, idf_file, feature_file, topn):
        super().__init__()
        # self.topn = 10
        # self.idf_file = "/Users/selenasong/Desktop/CLAMS/tfidf-kw-detection/tfidf_vectors.pkl"
        # self.feature_dict_file = "/Users/selenasong/Desktop/CLAMS/tfidf-kw-detection/features.pkl"

        self.topn = topn
        self.idf_file = idf_file
        self.feature_dict_file = feature_file


    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        # When using the ``metadata.py`` leave this do-nothing "pass" method here. 
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._annotate
        for arg, val in parameters.items():
            print("Parameter %s=%s" % (arg, val))
            # as we defined this `error` parameter in the app metadata
            if arg == 'error' and val is True:
                raise Exception("Exception - %s" % parameters['error'])

        # Initialize the MMIF object from the string if needed
        self.mmif = mmif if type(mmif) is Mmif else Mmif(mmif)

        # process the text documents in the documents list
        for doc in self.mmif.get_documents_by_type(DocumentTypes.TextDocument):
            new_view = self._new_view(parameters, doc.id)
            # _run_nlp_tool() is the method that does the actual work
            self._run_nlp_tool(doc, new_view, doc.id, self.idf_file, self.feature_dict_file)
        # return the MMIF object
        return self.mmif

    def _new_view(self, runtime_config, docid=None):
        view = self.mmif.new_view()
        view.metadata.app = self.metadata.identifier
        # first thing you need to do after creating a new view is "sign" the view
        # the sign_view() method will record the app's identifier and the timestamp
        # as well as the user parameter inputs. This is important for reproducibility.
        self.sign_view(view, runtime_config)
        # then record what annotations you want to create in this view
        # view.new_contain(Uri.DOCUMENT, document=docid)
        view.new_contain(AnnotationTypes.Alignment)
        return view

    def _run_nlp_tool(self, doc, new_view, full_doc_id, idf_file, feature_dict_file):
        """Run the NLP tool over the document and add annotations to the view, using the
        full document identifier (which may include a view identifier) for the document
        property."""
        text = doc.text_value

        # load idf values
        with open(idf_file, 'rb') as f:
            idf_values = pickle.load(f)

        # load feature dict
        with open(feature_dict_file, 'rb') as file:
            feature_dict = pickle.load(file)

        # restore the values
        cv = CountVectorizer(vocabulary=feature_dict)
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.idf_ = idf_values

        # get the feature names out into a list
        feature_names = feature_dict

        # get keywords for the document
        keywords = get_keywords(text, feature_names, self.topn, tfidf_transformer, cv)

        # create the document to store the keywords
        keywords_doc = new_view.new_textdocument(text=keywords)

        a = new_view.new_annotation(Uri.TEXT)

        a.add_property('document', full_doc_id)
        a.add_property('keywords_file', keywords_doc.id)

        # align the original text file and the keywords file
        new_view.new_annotation(AnnotationTypes.Alignment, source=full_doc_id, target=keywords_doc.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parser.add_argument("--idf_file", action="store", help="path to idf file")
    parser.add_argument("--feature_dict_file", action="store", help="path to feature dict file")
    parser.add_argument("--topn", action="store", type=int, default=10, help="top n keywords to return ")
    # add more arguments as needed

    parsed_args = parser.parse_args()

    # create the app instance
    app = TfidfKeywordDetector(parsed_args.idf_file, parsed_args.feature_dict_file, parsed_args.topn)

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
