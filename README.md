# Fingerprinting web servers through Transformer-encoded HTTP response headers (Darwinkel, 2023)
Code and datasets for my bachelor's thesis.

# License
The code of this project is licensed under the GNU GPLv3. The data that I have collected and processed (e.g. in the `data_*` folders) is licensed under the Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.

# Abstract
_We investigated the feasibility of using state-of-the-art deep learning, big data, and natural language processing techniques to improve the ability to detect vulnerable web server versions.
As having knowledge of specific version information is crucial for vulnerability analysis, we attempted to improve the accuracy and specificity of web server fingerprinting as compared to existing rule-based systems._

_To investigate this, we made various ambiguous and non-standard HTTP requests to 4.77 million domains, and collected HTTP response status lines for each request._
_We then trained a byte-level BPE tokenizer and RoBERTa encoder to represent these individual status lines through unsupervised masked language modelling._
_To represent the web server of a single domain, we dimensionality-reduced each encoded response line and concatenated them._
_A Random Forest and multilayer perceptron classified these encoded, concatenated samples._
_The classifiers achieved 0.94 and 0.96 macro f1-score respectively on detecting five of the most popular publicly available origin web servers, which make up roughly half of the World Wide Web._
_On the task of classifying 347 major type and minor version pairs, the multilayer perceptron achieved a weighted f1-score of 0.55._
_Analysis of Gini impurity suggests that our test cases are meaningful discriminants of web server types, and our high f1-scores are unprecedented and demonstrate that our proposed method is a viable alternative to traditional rule-based systems._
_In addition, this innovative method opens up avenues for future work, many of which will likely result in even greater classification performance._

# Instructions for replicating the experiment
The code was originally written in Python 3.10, guided by `isort`,`black`, `mypy`, `pylint`, and `vulture`. It's a bit of a mess, and the documentation is subpar. I haven't bothered to clean it properly as nobody will probably try to replicate my research. If you are interested in using the code and data, don't hesitate to contact me so I can help you get started. If desired, I can send my own processed files for reference.
You should probably read the thesis itself first.

## Creating the raw data
1. Run `collector.py` through `harbinger.sh` to create raw output files as `batch_.tsv`
2. concatenate generated batch files (including headers) with Unix utilities into a single file

## Cleaning and filtering the data
3. Clean the data and select viable samples with `preprocess_filter_by_bad_samples.py` from `concatenated_data.tsv` into `preprocessed_*.tsv`
   * `python3 code/create_dataset/preprocess_filter_by_bad_samples.py data_processed/concatenated_data_withheaders.tsv`
4. Filter the data by target classes with `preprocess_filter_by_target_labels.py` from `preprocessed.tsv` into `preprocessed_filtered.tsv`
   * `python3 code/create_dataset/preprocess_filter_by_target_labels.py preprocessed_.tsv`

## Preparing support data
5. Generate a list of unique values to generate a file to use for unsupervised training with `prepare_embeddings_list.py` from `concatenated_data.tsv` into `embeddings_list_.tsv`
   * `python3 code/create_support_data/prepare_embeddings_list.py data_processed/concatenated_data_withheaders.tsv`
6. Remove some redundant HTML-only lines from `embeddings_list.tsv` with Unix utilities
7. Generate a list of unique classes for HuggingFace
   * `python3 code/create_support_data/prepare_classes_lists.py preprocessed_filtered_.tsv`
 
## Training the Transformer encoder
8. Train a new tokenizer with `train_tokenizer.py` from an `embeddings_list.tsv`
   * `python3 code/create_embedding_model/train_tokenizer.py embeddings_list.tsv`
9. Train/fine-tune a (new) model with `train_embeddings_scratch.py` or `train_embeddings_finetune.py` from an `embeddings_list.tsv`
   * `python3 code/create_embedding_model/train_embeddings_.py embeddings_list.tsv`
10. Run `prepare_encoded_inputs.py` on `preprocessed_filtered.tsv` and `embeddings_list.tsv` to extract features from raw text columns, downsize them to 64 dimensions, and save them as a HuggingFace dataset
    * `python3 code/create_embedding_model/prepare_encoded_inputs.py preprocessed_filtered.tsv embeddings_list.tsv`

## Training and evaluating the classifiers
Run any classifier from `code/evaluate_embeddings` after `datasets/datasets/http-header-split-embedded-data-v1` has been created.

## Data analysis and plotting
See assorted files in `code/analyze_dataset`.

## Playing with the Transformer encoder
See assorted files in `code/play_with_model`.