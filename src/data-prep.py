import time
import traceback

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
    prev_request_time = 0

    def request(self, *args, timeout=None, **kwargs):
        """
        Wrap the request method with logging capabilities.
        """
        response = super().request(*args, **kwargs)

        print(f"Response from: {response.url}")

        curr_time = time.perf_counter()
        print(f"Time from previous request: {curr_time - self.prev_request_time}")
        self.prev_request_time = curr_time

        return response


# More info at https://praw.readthedocs.io/en/stable/getting_started/configuration/options.html
COMMENT_PREFIX = "t1_"
USER_PREFIX = "t2_"
SUBMISSION_PREFIX = "t3_"

# Dataframe constants
NO_USER = "N/A"
COMMENT_COL = "comment"
AUTHOR_COL = "author"
RESPONDING_TO_COL = "responding_to"

USER_COL = "user"
SENTIMENT_COL = "sentiment_score"

# Get data from up to two months ago
TWO_MONTHS_AGO = date.today() - timedelta(days=1)
THRESHOLD_DATE = TWO_MONTHS_AGO
OUT_FOLDER = "./output"


def main():
    # some settings (including oauth) imported from praw.ini
    reddit = praw.Reddit(site_name="wsb", requestor_class=LoggingRequestor)

    print("--------- Retrieving r/wsb... ---------")
    r_wsb = reddit.subreddit("wallstreetbets")

    print("--------- Fetching r/wsb comments... ---------")
    # Fetch and save comment data
    wsb_data = get_comment_data(r_wsb)

    print("--------- Saving r/wsb comments into csv... ---------")
    wsb_data.to_csv(f"{OUT_FOLDER}/wsb-2m-data-{1}.csv")

    print("--------- Calculating r/wsb users' sentiment... ---------")
    # Calculate and save user sentiment data
    user_sentiment_data = get_user_sentiment(wsb_data, author_col=AUTHOR_COL, comment_col=COMMENT_COL,
                                             user_col=USER_COL, sentiment_col=SENTIMENT_COL)

    print("--------- Saving r/wsb users' sentiment into csv... ---------")
    user_sentiment_data.to_csv(f"{OUT_FOLDER}/wsb-2m-user-sentiment-{1}")

    print("--------- Completed ---------")


def get_comment_data(subreddit: praw.models.Subreddit) -> pd.DataFrame:
    subreddit_data = pd.DataFrame()
    try:
        # Create a submission iterator and get submissions from the past two months.
        # From those, extract the comments and put them into a dataframe
        submissions_iterator = subreddit.stream.submissions()
        at_least_one = False
        counter = 0
        for submission in submissions_iterator:
            counter += 1
            print(f"[Submission #{counter}]")
            sub_date = datetime.utcfromtimestamp(submission.created_utc).date()

            # Fetch only submission within (after) the threshold date
            if sub_date > THRESHOLD_DATE:
                sub_data = fetch_data_from_submission(submission,
                                                      author_col=AUTHOR_COL,
                                                      comment_col=COMMENT_COL,
                                                      responding_to_col=RESPONDING_TO_COL)
                subreddit_data.append(sub_data)
                at_least_one = True
            else:
                if not at_least_one:
                    # Skip until you get a submission within the threshold date
                    print("Skipping submission because it is out of date interval...")
                    continue
                else:
                    # If at least one submission has been fetched, and we fall here,
                    # it means the iterator has begun a new batch of submissions, which
                    # we do not want since it means that all those submissions are going
                    # to be beyond the threshold date
                    break
    except Exception as e:  # TODO debug
        print("Error caught: ")
        print(traceback.format_exc())
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
    print(f"Fetching submission '{submission.title}', created at: {creation_date}")
    print(f"(Date interval: {TWO_MONTHS_AGO} - {date.today()})")

    # Treat submission as if it was a regular comment
    sub_author_id = get_author_id_safe(submission)
    authors.append(sub_author_id)

    comments.append(f"{submission.title} - {submission.selftext}")
    responding_to.append(NO_USER)  # Submission is in response to no one

    sub_id = remove_prefixes(submission.id)
    comment_authors[sub_id] = sub_author_id

    # Submission is the root so it has no parent.
    # Set the parent to itself to avoid errors when converting
    # from parent id to author id
    parent_ids.append(sub_id)

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
        progress_pct = round(((comment_counter / tot_comments) * 100))
        print(f"Fetching comment #{comment_counter} of max {tot_comments} [{progress_pct}%]")

        # Add relevant comment data to the lists
        # Author might be deleted, in which case it is None
        author_id = get_author_id_safe(c)
        authors.append(author_id)

        comments.append(c.body)

        # Store data used to compute the "responding_to" column
        # If the comment is top level, the id in parent_id is the id of the submission
        parent_id = remove_prefixes(c.parent_id)
        parent_ids.append(parent_id)

        c_id = remove_prefixes(c.id)
        comment_authors[c_id] = author_id

    try:
        df_data = {author_col: authors, comment_col: comments,
                   responding_to_col: [comment_authors[p_id] for p_id in parent_ids]}
    except Exception as e:  # TODO debug
        raise e
    return pd.DataFrame(data=df_data)


def get_author_id_safe(praw_obj) -> str:
    """
    Function that handles the case where the author of something (comment, submission, etc.)
    has been deleted, in which case it is represented as None by PRAW.
    Returns a string id, different from None, and net of reddit api prefixes.
    :param praw_obj: praw object such as comment or submission
    :return: author id or NO_USER
    """
    if praw_obj.author is None:
        return NO_USER
    else:
        return remove_prefixes(praw_obj.author.id)


def remove_prefixes(obj_id: str) -> str:
    """
    Used to extract the actual id, net of prefixes, from a given id string
    :param obj_id: id of a praw object
    :return: id, net of reddit api prefixes
    """
    temp = obj_id.replace(COMMENT_PREFIX, "")
    temp = temp.replace(USER_PREFIX, "")
    temp = temp.replace(SUBMISSION_PREFIX, "")

    return temp


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

    if len(comment_data) == 0:
        print("Comment data is an empty dataframe")
        return pd.DataFrame()

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
    main()
