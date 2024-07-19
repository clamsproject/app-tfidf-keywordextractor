
# TFIDF Keyword Extractor 

## Description

This app extracts keywords in a text document according to tokens' TF-IDF scores. The IDF scores are generated from 
a given list of text files in a directory.

## Information on the available model
The current available model for keyword extraction is trained with 22 out of 24 NewsHour transcripts listed in
[batch2.txt](https://github.com/clamsproject/aapb-annotations/blob/9cbe41aa124da73a0158bfc0b4dbf8bafe6d460d/batches/batch2.txt).
Excluded files' names and reasons of exclusion are:
* `cpb-aacip-525-028pc2v94s`: File not found in the dataset
* `cpb-aacip_507-r785h7cp0z`: Contains no transcript but an error message

This model is trained with English stopwords removed. 
Tokens that appears in more than 85% of these 22 documents are also removed (i.e., `max_df=0.85`) 

## User instruction
### System requirements
* Requires Python3 with `clams-python`, `clams-utils` and `scikit-learn` to run the app locally.
* Requires an HTTP client utility (such as `curl`) to invoke and execute analysis.
* Requires docker to run the app in a Docker container 

Run `pip install -r requirements.txt` to install the requirements.

### Train a model with NewsHour transcripts using `tfidf.py`
> **NOTE:**
> If you only look to use the keyword extractor app instead of training your own model, 
> please skip this section and follow instructions in the next section. 

After getting into the working directory, run the following line on the target dataset:

`python tfidf.py --dataPath path/to/target/dataset/directory`

By running this line, `tfidf.py` does 2 things:
* cleans all transcripts in a given directory.
* generates a pickle file named `idf_feature_file.pkl` by default that stores the IDF values and the corresponding 
feature dictionary. Currently, this file is not allowed to be renamed, or it affects running `cli.py` later on. 

~~If the pickle file needs to be named differently from the default, then add
`--idfFeatureFile` and the expected name to the command above to change the names.~~ 

> **~~warning:~~**
> ~~renaming the pickle file at this step will affect the command for running the keyword extractor in the later step~~

Default value for max document frequency is 0.85. If a different value for is required, then add `--maxDf` 
and the expected float value (max value is 1.0) to the command above. 

### Extract keywords using TF-IDF values

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

To run this app in CLI:

`python cli.py --optional_params <input_mmif_file_path> <output_mmif_file_path>`

2 types of input `MMIF` files are acceptable here:
* The ones that are generated through `clams source text:/path/to/the/target/txt/file` to extract keywords for a single
text document.
* The ones whose last view containing TextDocument(s) is the view to extract keywords from.

> **~~note:~~**
> ~~If the file storing the IDF values and the feature dict do not have the default file name as listed above, 
> then when running `cli.py`according to the Apps documentation, before entering input and output `MMIF` files, 
> add `--idfFeatureFile` and the corresponding file name~~ 

Default number of keywords extracted from a given text document is 10. If the number of extracted keywords is required 
to be different from 10, when running `cli.py`, add `--topN` and a corresponding integer value. 

Two scenarios may be seen if the input text document is too short:
1. If the number of tokens in a text document is smaller than the value of `topN`, 
then no keywords will be extracted. 
2. If the text contains lots of stopwords, then the number of extracted keywords can be less than the value of `topN`,
because the app ignores all stopwords when finding keywords. 

### Configurable runtime parameter

For the full list of parameters, please refer to the app metadata from the [CLAMS App Directory](https://apps.clams.ai) 
or the [`metadata.py`](metadata.py) file in this repository.

