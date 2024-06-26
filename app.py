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

# Import functions from tfidf.py
from tfidf import get_keywords

class TfidfKeywordextractor(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory.
        # When using the ``metadata.py`` leave this do-nothing "pass" method here.
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        # Initialize the MMIF object from the string if needed
        self.mmif = mmif if type(mmif) is Mmif else Mmif(mmif)

        # process the text documents in the documents list
        for doc in self.mmif.get_documents_by_type(DocumentTypes.TextDocument):
            idf_feature_file = parameters['idfFeatureFile']
            topn = parameters['topN']
            new_view = self._new_view(parameters)
            # _run_nlp_tool() is the method that does the actual work
            self._keyword_extractor(doc, new_view, doc.long_id, idf_feature_file, topn)
        # return the MMIF object
        return self.mmif

    def _new_view(self, runtime_config):
        view = self.mmif.new_view()
        view.metadata.app = self.metadata.identifier
        # first thing you need to do after creating a new view is "sign" the view
        # the sign_view() method will record the app's identifier and the timestamp
        # as well as the user parameter inputs. This is important for reproducibility.
        self.sign_view(view, runtime_config)
        # then record what annotations you want to create in this view
        view.new_contain(DocumentTypes.TextDocument)
        view.new_contain(AnnotationTypes.Alignment)
        return view

    def _keyword_extractor(self, doc, new_view, full_doc_id, idf_feature_file, topn):
        """Run the NLP tool over the document and add annotations to the view, using the
        full document identifier (which may include a view identifier) for the document
        property."""
        text = doc.text_value

        # load idf values
        with open(idf_feature_file, 'rb') as f:
            idf_feature_values = pickle.load(f)
            idf_values = idf_feature_values["idf_values"]
            feature_dict = idf_feature_values["feature_dict"]

        # restore the values
        cv = CountVectorizer(vocabulary=feature_dict)
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.idf_ = idf_values

        # get keywords for the document
        keywords_dict = get_keywords(text, feature_dict, topn, tfidf_transformer, cv)
        keywords = ""
        tfidf_values = ""
        for keyword, tfidf_value in keywords_dict.items():
            keywords += keyword + " "
            tfidf_values += str(tfidf_value) + " "

        # create the document to store the keywords
        keywords_doc = new_view.new_textdocument(text=keywords.strip())
        keywords_doc.add_property('scores', tfidf_values.strip())

        a = new_view.new_annotation(AnnotationTypes.Alignment, source=full_doc_id, target=keywords_doc.long_id)

        a.add_property('document', full_doc_id)
        a.add_property('keywords_file', keywords_doc.long_id)


def get_app():
    """
    This function effectively creates an instance of the app class, without any arguments passed in, meaning, any
    external information such as initial app configuration should be set without using function arguments. The easiest
    way to do this is to set global variables before calling this.
    """
    # for example:
    # return TfidfKeywordextractor(create, from, global, params)
    return TfidfKeywordextractor()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    # if get_app() call requires any "configurations", they should be set now as global variables
    # and referenced in the get_app() function. NOTE THAT you should not change the signature of get_app()
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
