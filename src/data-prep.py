import praw
import praw.models
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from prawcore import Requestor
from datetime import datetime, date, timedelta

# Needed for vader sentiment analyzer
nltk.download("vader_lexicon")


class LoggingRequestor(Requestor):
    def request(self, *args, timeout=None, **kwargs):
        """
        Wrap the request method with logging capabilities.
        """
        response = super().request(*args, **kwargs)

        print(f"Response from: {response.url}")

        return response


# More info at https://praw.readthedocs.io/en/stable/getting_started/configuration/options.html
COMMENT_PREFIX = "t1_"
USER_PREFIX = "t2_"
SUBMISSION_PREFIX = "t3_"

# Dataframe constants
NO_USER = "N/A"  # used in the responding to column of submission records
COMMENT_COL = "comment"
AUTHOR_COL = "author"
RESPONDING_TO_COL = "responding_to"

USER_COL = "user"
SENTIMENT_COL = "sentiment_score"

# Get data from up to two months ago
TWO_MONTHS_AGO = date.today() - timedelta(days=60)
THRESHOLD_DATE = TWO_MONTHS_AGO
OUT_FOLDER = "./output"


def get_comment_data(subreddit: praw.models.Subreddit) -> pd.DataFrame:
    subreddit_data = pd.DataFrame()
    try:
        # Create a submission iterator and get submissions from the past two months.
        # From those, extract the comments and put them into a dataframe
        submissions_iterator = subreddit.stream.submissions()
        for submission in submissions_iterator:
            sub_date = datetime.utcfromtimestamp(submission.created_utc)

            # TODO first test with 2 days ago to see if i'm doing stuff right
            THRESHOLD_DATE = datetime.today() - timedelta(days=2)

            if sub_date > THRESHOLD_DATE:
                sub_data = fetch_data_from_submission(submission,
                                                      author_col=AUTHOR_COL,
                                                      comment_col=COMMENT_COL,
                                                      responding_to_col=RESPONDING_TO_COL)
                subreddit_data.append(sub_data)
            else:
                # All the data has been fetched
                break
    except Exception as e:
        print(e)
    finally:
        # Make sure to return all the data fetched to this point, so it
        # doesn't have to be downloaded again (takes a long time)
        return subreddit_data


def fetch_data_from_submission(submission: praw.models.Submission,
                               author_col: str, comment_col: str,
                               responding_to_col: str) -> pd.DataFrame:
    """
    Fetch all the comments from a given submission and return a dataframe containing said comments.
    Title + selftext + author of the provided submission are treated as a comment in response to no user.

    :param submission: reddit submission to extract comments from
    :param author_col: name of the author column of the returned dataframe
    :param comment_col: name of the comment texts column of the returned dataframe
    :param responding_to_col: name of the column of the returned dataframe that contains the user
            to which a certain comment is responding
    :return: dataframe containing comment data
    """

    # Build the dataframe containing the data fetched from the submissions
    authors = []
    comments = []
    responding_to = []

    # used to pair the comment id with its author.
    # this is useful to understand who a certain comment is responding to,
    # since comment.parent_id points to the id of a comment and not a redditor
    comment_authors = {}

    # List containing id of parent comments
    # Used later in conjunction with the above dictionary to retrieve
    # the author id and ultimately populate the column that contains
    # the user a certain comment is responding to
    parent_ids = []

    # Log some data to the console to keep track of progress
    unix_time_date = submission.created_utc
    creation_date = datetime.utcfromtimestamp(unix_time_date).date()
    print(f"Fetching submission '{submission.id}' data, created at: {creation_date}")
    print(f"(Date interval: {TWO_MONTHS_AGO} - {date.today()})")

    # Treat submission as if it was a regular comment
    authors.append(submission.author.id)
    comments.append(f"{submission.title} - {submission.selftext}")
    responding_to.append(NO_USER)  # Submission is in response to no one
    comment_authors[submission.id] = submission.author.id

    # Using replace_more is like pressing an all the buttons that say "load more comments"
    # With the limit set to None, it tells praw to basically fetch all the comments, since,
    # by default, not all of them are loaded
    # See https://praw.readthedocs.io/en/latest/tutorials/comments.html for more info
    submission.comments.replace_more(limit=None)

    # tot_comments might not be exact because it also counts deleted comments, blocked comments, etc
    comment_counter = 0
    tot_comments = submission.num_comments
    for c in submission.comments.list():
        # Progress logging
        comment_counter += 1
        progress_pct = (comment_counter / tot_comments) * 100
        print(f"Fetching comment #{comment_counter} of ~{tot_comments} [{progress_pct}%]")

        # Add relevant comment data to the lists
        author_id = c.author.id
        authors.append(author_id)

        comments.append(c.body)

        # Store data used to compute the "responding_to" column
        # If the comment is top level, the id in parent_id is the id of the submission
        parent_id = c.parent_id if SUBMISSION_PREFIX not in c.parent_id else submission.author.id
        parent_ids.append(parent_id)
        comment_authors[c.id] = author_id

    df_data = {author_col: authors, comment_col: comments,
               responding_to_col: [comment_authors[p_id] for p_id in parent_ids]}
    return pd.DataFrame(data=df_data)


def get_user_sentiment(comment_data: pd.DataFrame,
                       author_col: str, comment_col: str,
                       user_col: str, sentiment_col: str) -> pd.DataFrame:
    """
    Calculates the sentiment of users based on their comments.
    Returns a dataframe containing all user's sentiment data.
    :param comment_data: dataframe containing comment data (authors, comment texts)
    :param author_col: name of the column in the provided dataframe
            that contains the authors of the comments
    :param comment_col: name of the column in the provided dataframe that contains
            the actual comment text
    :param user_col: name of the column in the output dataframe that contains
            the id of each user
    :param sentiment_col: name of the column in the output dataframe that contains
            the sentiment score for each user
    :return: pandas dataframe containing user sentiment data
    """

    # https://github.com/cjhutto/vaderSentiment#resources-and-dataset-descriptions
    # positive sentiment: compound score >= 0.05
    # neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    # negative sentiment: compound score <= -0.05
    sentiment_analyzer = SentimentIntensityAnalyzer()
    data = comment_data.copy()

    # Calculate sentiment score for each comment
    # only compound score is necessary because it sums up neu, pos and neg scores.
    text_data = preprocess_comments(data[comment_col])
    data[sentiment_col] = text_data.apply(lambda x: (sentiment_analyzer.polarity_scores(x))["compound"])

    # Calculate the sentiment of each user and keep only the necessary columns
    user_sentiment = data.groupby(by=[author_col], as_index=False).mean()
    user_sentiment = user_sentiment[[author_col, sentiment_col]]
    user_sentiment = user_sentiment.rename({author_col: user_col}, axis="columns")

    return user_sentiment


def preprocess_comments(comments: pd.Series) -> pd.Series:
    """
    Performs cleaning and preprocessing operations on the provided comments
    :param comments:
    :return: preprocessed comments
    """
    # Set everything to lower case to make comparisons easier
    comments = comments.str.lower()

    # Remove urls TODO risolvere url e analizzare testo?
    comments = comments.apply(lambda x: re.sub(r"http\S+", "", x))

    # Remove all the special characters (keep only word characters)
    comments = comments.apply(lambda x: ' '.join(re.findall(r'\w+', x)))

    # remove all single characters
    comments = comments.apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', x))

    # Substituting multiple spaces with single space
    comments = comments.apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.IGNORECASE))

    return comments


if __name__ == "__main__":
    # some settings (including oauth) imported from praw.ini
    reddit = praw.Reddit(site_name="wsb", requestor_class=LoggingRequestor,
                         comment_kind=COMMENT_PREFIX, redditor_kind=USER_PREFIX,
                         submission_kind=SUBMISSION_PREFIX)

    print("--------- Retrieving r/wsb... ---------")
    r_wsb = reddit.subreddit("wallstreetbets")

    print("--------- Fetching r/wsb comments... ---------")
    # Fetch and save comment data
    wsb_data = get_comment_data(r_wsb)

    print("--------- Saving r/wsb comments into csv... ---------")
    wsb_data.to_csv(f"{OUT_FOLDER}/wsb-2m-data-{datetime.now()}.csv")

    print("--------- Calculating r/wsb users' sentiment... ---------")
    # Calculate and save user sentiment data
    user_sentiment_data = get_user_sentiment(wsb_data, author_col=AUTHOR_COL, comment_col=COMMENT_COL,
                                             user_col=USER_COL, sentiment_col=SENTIMENT_COL)

    print("--------- Saving r/wsb users' sentiment into csv... ---------")
    user_sentiment_data.to_csv(f"{OUT_FOLDER}/wsb-2m-user-sentiment-{datetime.now()}")

    print("--------- Completed ---------")
