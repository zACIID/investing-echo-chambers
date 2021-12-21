import time
import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, date, timedelta
from interactions import Interaction, SubredditInteractions
from constants import USER_COL, TEXT_COL, INTERACTED_WITH_COL, SENTIMENT_COL

# Needed for vader sentiment analyzer
nltk.download("vader_lexicon")

# Get data from up to two months ago
FROM_DATE = datetime.today() - timedelta(days=1)
TO_DATE = datetime.today()

OUT_FOLDER = "./output"


def main():
    print("--------- Initializing r/wsb... ---------")
    wsb = SubredditInteractions(subreddit="wallstreetbets", ini_site_name="wsb",
                                date_after=FROM_DATE, date_before=TO_DATE,
                                logger=print)

    print("--------- Fetching r/wsb interactions... ---------")
    # Fetch and save comment data
    wsb_interactions = wsb.fetch_interactions()
    print("--------- r/wsb interactions fetched ---------")

    # Create and save interactions dataframe
    interactions_df = get_interaction_df(interactions=wsb_interactions,
                                         user_out_col=USER_COL,
                                         text_out_col=TEXT_COL,
                                         interacted_with_out_col=INTERACTED_WITH_COL)
    if len(interactions_df) > 0:
        print("--------- Saving r/wsb interactions into csv... ---------")
        interactions_df.to_csv(f"{OUT_FOLDER}/wsb-interactions-{date.today()}.csv")

    # Calculate and save user sentiment data
    print("--------- Calculating r/wsb users' sentiment... ---------")
    user_sentiment_df = get_user_sentiment_df(interactions=wsb_interactions,
                                              user_out_col=USER_COL,
                                              sentiment_out_col=SENTIMENT_COL)
    if len(user_sentiment_df) > 0:
        print("--------- Saving r/wsb users' sentiment into csv... ---------")
        user_sentiment_df.to_csv(f"{OUT_FOLDER}/wsb-2m-user-sentiment-{date.today()}.csv")

    print("--------- Completed ---------")


def get_interaction_df(interactions: list[Interaction],
                       user_out_col: str, text_out_col: str,
                       interacted_with_out_col: str) -> pd.DataFrame:
    """
    Returns a dataframe based on the provided interactions.
    :param interactions: list of user interactions
    :param user_out_col: name of the column, in the output dataframe,
        that contains all the authors of the fetched text
    :param text_out_col: name of the column, in the output dataframe,
        that contains all the comment and submission text
    :param interacted_with_out_col: name of the column, in the output dataframe,
        that contains the users that the fetched text was in response to.
        For submissions, the user is treated as if it was responding to itself.
    :return:
    """

    if len(interactions) == 0:
        print("Interaction list is empty")
        return pd.DataFrame()

    interaction_data = {
        user_out_col: [i.user for i in interactions],
        text_out_col: [i.text_data for i in interactions],
        interacted_with_out_col: [i.interacted_with for i in interactions]
    }

    return pd.DataFrame(data=interaction_data)


# TODO refactor sentiment methods in a class that accepts interactions data and returns Sentiment objects (user + score)
#       this way, the conversion to dataframe is done only here in app.py
#       Class name might be SentimentProcessor
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


def test_main():
    from src.interactions import LoggingRequestor
    import praw

    # settings imported from praw.ini
    reddit = praw.Reddit(site_name="wsb", requestor_class=LoggingRequestor)
    r_wsb = reddit.subreddit("wallstreetbets")

    print("----------- Fetching submissions -----------")
    counter = 0
    submissions_iterator = r_wsb.new(limit=None)
    for submission in submissions_iterator:
        counter += 1
        print(f"---------- Submission #{counter} ----------")
        print(submission.title)
        print(submission.selftext)
        print(submission.created_utc)

        print("----------- Fetching comments -----------")
        counter_c = 0
        submission.comments.replace_more(limit=0)
        comments_iterator = submission.comments.list()
        start_time = time.perf_counter()
        for comment in comments_iterator:
            curr_time = time.perf_counter()
            counter_c += 1
            print(f"Comment #{counter_c} - time: {curr_time - start_time}")
            print(f"Autore: {comment.author}")
            print(comment.id)
            print(comment.parent_id)
            # print(comment.body)

            start_time = curr_time


if __name__ == "__main__":
    main()
    # test_main()
