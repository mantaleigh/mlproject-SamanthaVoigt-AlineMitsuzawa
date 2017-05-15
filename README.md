# Machine Learning Project: Predicting Ideal Airbnb Listing Prices from Host Supplied Information

Samantha Voigt and Aline Mitsuzawa
[Paper Link](https://docs.google.com/a/wellesley.edu/document/d/1Qf_47n5UwrwC3flkg5XUxZr_613aAzPPt3Z6RFJi6lA/edit?usp=sharing)

## Download Datasets

[Dropbox Link](https://www.dropbox.com/sh/ntvby0v6gdsrchk/AADMHN_4u18vPW0zZbw6iFvza?dl=0)

All dataset files come from [Inside Airbnb](http://www.insideairbnb.com).

Note that model.py expects these .csv files to be located inside a /data/raw_data folder within the project.

## How To Run

*All code is in model.py*

If you are running a one-off model:

* `run_model(max_price=500, text_type_vec='count', text_max_feats=500, plot=False, train_size=0.85)`

If you wish to run a grid search:

Simply run `grid_search()`

## Grid Search Output

Find output of a previously-run grid search of hyperparameters within grid_search_output.txt
