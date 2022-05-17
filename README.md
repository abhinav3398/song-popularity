song popularity
==============================

In this project we try to assess whether it is possible to get some relevent insight from external features of songs, such as acousticness, tempo, loudness, instrumentalness, song duration, etc. in order to predict song popularity.

The project is organizes as followes
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Problem Statement
we will try to answerthese Business Questions:

1. Is it possible to predict whether a song would be popular or not, only from external characteristics of the song and metadata, such as, Instrumentalness, Danceability, Loudness, Tempo, etc.?

2. Which Factors would help the most in predicting song popularity and which features we can remove in further analysis?

3. How much information does the provided dataset have? i.e. With how much accuracy would we able to predict the popularity of a given song?

For example, If we have less data then what other information would we need to improve the predictions?

## Dataset Description
This work uses dataset from recent [Kaggle competition "Song Popularity Prediction”](https://www.kaggle.com/c/song-popularity-prediction) which is an exhaustive collection of audio features and metadata for about 50,000 songs. The audio features include attributes about the music track itself, such as song duration, key, audio mode, time signature, The metadata uses more abstract features, such as danceability, energy, instrumentalness, liveness, etc. This Dataset is a subset which was derived from Spotify web api which provides access to user related data, like playlists and music that the user saves in their Music library containing data regarding millions of songs and continuously being updated with new songs and changing user preference (Web API, n.d.)

**The Methodology, Analysis, and Modeling of the project can be found in the the [notebooks](notebooks) as well as in th [project report](reports/Capstone-Project-Report.docx).**

## Conclusion
We found that, we need more information about the song being classified as even using the best feature transformations and sophisticated models we weren't able able to get a relevent accuracy in order for the model to be reliably predict. As we have used most of the external information about the song, in our modeling and still got unsatisfied score, we will need internal characteristics of the song such as song lyrics or the actual audio data or at least portion of the song being predicted.

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
