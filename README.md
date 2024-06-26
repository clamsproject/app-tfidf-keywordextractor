
# TFIDF Keyword Extractor 

## Description

This app extracts keywords in a text document according to tokens' TF-IDF scores. The IDF scores are generated from 
a given list of text files in a directory.

## User instruction
### System requirements
* Requires Python3 with `clams-python`, and `scikit-learn` to run the app locally.
* Requires an HTTP client utility (such as `curl`) to invoke and execute analysis.
* Requires docker to run the app in a Docker container 

Run `pip install -r requirements.txt` to install the requirements.

### Generate IDF scores for tokens in text documents in a directory 
After getting into the working directory, run the following line on the target dataset:

`python tfidf.py --dataPath path/to/target/dataset/directory`

By running this line, tfidf.py generates a pickle file named `idf_feature_file.pkl` by default that stores the IDF values
and the corresponding feature dictionary for the use of later keyword extraction. 

If these files need to be named differently from the default, then add `--idfFeatureFile` and the expected
file name to the command above to change the names of the generated files. 

> **warning:**
> renaming files at this step will affect the command for running the keyword extractor in the later step 

Default value for max document frequency is 0.85. If a different value for is required, then add `--maxDf` 
and the expected float value (max value is 1.0) to the command above. 

### Extract keywords using TF-IDF values

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

> **note:**
> If the file storing the IDF values and the feature dict do not have the default file name as listed above, 
> then when running `cli.py`according to the Apps documentation, before entering input and output `mmif` files, 
> add `--idfFeatureFile` and the corresponding file name 

Default number of keywords extracted from a given text document is 10. If this number
is required to be different, when running `cli.py`, add `--topN` and a corresponding integer value

### Configurable runtime parameter

For the full list of parameters, please refer to the app metadata from the [CLAMS App Directory](https://apps.clams.ai) 
or the [`metadata.py`](metadata.py) file in this repository.

