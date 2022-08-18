import re

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.interactions import Interaction

# Needed for vader sentiment analyzer
nltk.download("vader_lexicon")


def get_user_sentiment_df(interactions: list[Interaction],
                          user_out_col: str, sentiment_out_col: str) -> pd.DataFrame:
    """
    Calculates the sentiment of users based on the provided interactions.
    Returns a dataframe containing all the involved users' sentiment data.
    :param interactions: list of user interactions
    :param user_out_col: name of the column, in the output dataframe,
        that contains the users that the sentiment score is calculated for.
    :param sentiment_out_col: name of the column in the output dataframe that contains
            the sentiment score for each user
    :return:
    """

    if len(interactions) == 0:
        print("Interaction list is empty")
        return pd.DataFrame()

    text_col = "text"
    user_data = {
        text_col: [i.text_data for i in interactions],
        user_out_col: [i.user for i in interactions]
    }
    user_df = pd.DataFrame(data=user_data)

    # Performance optimization: some rows might be duplicated, for
    # example if a bot posts the same text over and over, and keeping
    # them makes the merging operation quite heavier
    user_df = user_df.drop_duplicates()

    sentiment_df = get_text_sentiment_df(interactions=interactions,
                                         text_out_col=text_col,
                                         sentiment_out_col=sentiment_out_col)

    # Calculate the sentiment of each user (average of their texts)
    user_sentiment_df = user_df.merge(sentiment_df, on=[text_col])
    user_sentiment_df = user_sentiment_df.groupby(by=[user_out_col], as_index=False).mean()

    # Text column here is not needed, just user and their sentiment
    user_sentiment_df = user_sentiment_df[[user_out_col, sentiment_out_col]]

    return user_sentiment_df


def get_text_sentiment_df(interactions: list[Interaction],
                          text_out_col: str, sentiment_out_col: str) -> pd.DataFrame:
    if len(interactions) == 0:
        print("Interaction list is empty")
        return pd.DataFrame()

    user_data = {
        text_out_col: [i.text_data for i in interactions]
    }
    text_df = pd.DataFrame(data=user_data)

    # https://github.com/cjhutto/vaderSentiment#resources-and-dataset-descriptions
    # positive sentiment: compound score >= 0.05
    # neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    # negative sentiment: compound score <= -0.05
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # Calculate sentiment score for each text
    # It's important to preprocess it to make analysis more accurate
    # Only compound score is necessary because it sums up neu, pos and neg scores.
    preprocessed_col = "preprocessed_text"
    text_df[preprocessed_col] = _preprocess_text(text_df[text_out_col])
    text_df[sentiment_out_col] = text_df[preprocessed_col].apply(
        lambda txt: (sentiment_analyzer.polarity_scores(txt))["compound"])

    # The preprocessed col doesn't need to be returned,
    # it was useful for sentiment analysis only
    return text_df[[text_out_col, sentiment_out_col]]


def _preprocess_text(text_data: pd.Series) -> pd.Series:
    """
    Performs cleaning and preprocessing operations on the provided text data.
    :param text_data:
    :return: preprocessed text
    """
    # Set everything to lower case to make comparisons easier
    text_data = text_data.str.lower()

    # Remove urls
    text_data = text_data.apply(lambda x: re.sub(r"http\S+", "", x))

    # Remove all the special characters (keep only word characters)
    text_data = text_data.apply(lambda x: ' '.join(re.findall(r'\w+', x)))

    # Replace all single characters with space (so words don't get glued together)
    text_data = text_data.apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))

    # Substituting multiple spaces with single space
    text_data = text_data.apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.IGNORECASE))

    return text_data
