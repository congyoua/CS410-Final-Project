Traditional_Method.ipynb: This notebook contains code for our Vector Space Model (VSM) and associated data analysis & results. It does not contain a predict pipleline, so please modify the notebook if you wish to test your own samples.

ML_Method.ipynb: This notebook provides preprocessing and training code for our DistillBERT based model, along with data analysis & results. It also includes training curves, metrics table, confusion matrix as well as successful and unsuccessful prediction examples.

Predict(ML).py: This script runs the prediction pipeline for our ml model. To view available commands, type "python Predict(ML).py --help". For predicting on a tweet, use python "Predict(ML).py --tweet 'YOUR TWEET'".

requirements.txt: This file lists all Python dependencies necessary to run the notebooks. Please install all dependencies by executing "pip install -r requirements.txt". If you only intend to try the prediction pipeline, installing the torch and transformers libraries will suffice.





