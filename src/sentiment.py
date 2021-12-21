import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from interactions import Interaction

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

    user_col = "user"
    text_col = "text"
    user_data = {
        user_col: [i.user for i in interactions],
        text_col: [i.text_data for i in interactions]
    }
    user_df = pd.Dataframe(data=user_data)

    # https://github.com/cjhutto/vaderSentiment#resources-and-dataset-descriptions
    # positive sentiment: compound score >= 0.05
    # neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    # negative sentiment: compound score <= -0.05
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # Calculate sentiment score for each comment
    # only compound score is necessary because it sums up neu, pos and neg scores.
    text_data = preprocess_comments(user_df[text_col])
    user_df[sentiment_out_col] = text_data.apply(lambda txt: (sentiment_analyzer.polarity_scores(txt))["compound"])

    # Calculate the sentiment of each user (average of their texts)
    user_sentiment = user_df.groupby(by=[user_col], as_index=False).mean()

    # Only user and sentiment columns are needed
    user_sentiment = user_sentiment[[user_col, sentiment_out_col]]
    user_sentiment = user_sentiment.rename({user_col: user_out_col}, axis="columns")

    return user_sentiment


def preprocess_comments(comments: pd.Series) -> pd.Series:
    """
    Performs cleaning and preprocessing operations on the provided comments
    :param comments:
    :return: preprocessed comments
    """
    # Set everything to lower case to make comparisons easier
    comments = comments.str.lower()

    # Remove urls
    comments = comments.apply(lambda x: re.sub(r"http\S+", "", x))

    # Remove all the special characters (keep only word characters)
    comments = comments.apply(lambda x: ' '.join(re.findall(r'\w+', x)))

    # remove all single characters
    comments = comments.apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', x))

    # Substituting multiple spaces with single space
    comments = comments.apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.IGNORECASE))

    return comments
