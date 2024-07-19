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
from mmif import Mmif, AnnotationTypes, DocumentTypes
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Import necessary function(s) from tfidf.py
from tfidf import get_keywords

class TfidfKeywordextractor(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.mmif = mmif if type(mmif) is Mmif else Mmif(mmif)

        # TODO: change the value of this parameter after the modelName parameter in metadata.py is solved
        idf_feature_file = './idf_feature_file.pkl'
        topn = parameters['topN']

        text_slicer_checker = self.mmif.get_view_contains(DocumentTypes.TextDocument)
        new_view = self._new_view(parameters)
        if text_slicer_checker is None:
            # scenario 1: single text document input.
            for doc in self.mmif.get_documents_by_type(DocumentTypes.TextDocument):
                self._keyword_extractor(doc, new_view, doc.long_id, idf_feature_file, topn)
        else:
            # scenario 2: input document is the one generated from the text slicer.
            docs = text_slicer_checker.get_annotations(DocumentTypes.TextDocument)
            for doc in docs:
                text = doc.text_value
                if len(text.split()) > topn:
                    self._keyword_extractor(doc, new_view, doc.long_id, idf_feature_file, topn)

        # return the MMIF object
        return self.mmif

    def _new_view(self, runtime_config):
        view = self.mmif.new_view()
        view.metadata.app = self.metadata.identifier
        self.sign_view(view, runtime_config)
        view.new_contain(DocumentTypes.TextDocument, text="keywords", scores="tfidf values")
        view.new_contain(AnnotationTypes.Alignment)
        return view

    def _keyword_extractor(self, doc, new_view, full_doc_id, idf_feature_file, topn):
        """Run the keyword extractor over the document and add annotations to the view, using the
        full document identifier (which may include a view identifier) for the document
        property."""
        text = doc.text_value

        # load idf values and feature dict
        with open(idf_feature_file, 'rb') as f:
            idf_feature_values = pickle.load(f)
            idf_values = idf_feature_values["idf_values"]
            feature_dict = idf_feature_values["feature_dict"]

        # restore the TfidfTransformer
        cv = CountVectorizer(vocabulary=feature_dict, stop_words='english')
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.idf_ = idf_values

        # get keywords and their corresponding tfidf scores for the document
        keywords_dict = get_keywords(text, feature_dict, topn, tfidf_transformer, cv)
        keywords = ""
        tfidf_values = []
        for keyword, tfidf_value in keywords_dict.items():
            keywords += keyword + " "
            tfidf_values.append(tfidf_value)

        # create the document to store the keywords and their tfidf values
        keywords_doc = new_view.new_textdocument(text=keywords.strip(), scores=tfidf_values)

        # create the alignment between the target document and the corresponding keywords and tfidf values
        new_view.new_annotation(AnnotationTypes.Alignment, source=full_doc_id, target=keywords_doc.long_id)


def get_app():
    """
    This function effectively creates an instance of the app class, without any arguments passed in, meaning, any
    external information such as initial app configuration should be set without using function arguments. The easiest
    way to do this is to set global variables before calling this.
    """
    return TfidfKeywordextractor()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
