import praw
import praw.models
import pandas as pd
from prawcore import Requestor
from datetime import datetime, date, timedelta


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

# Get data from up to two months ago
TWO_MONTHS_AGO = date.today() - timedelta(days=60)
THRESHOLD_DATE = TWO_MONTHS_AGO
OUT_FOLDER = "../output"


def main():
    # some settings (including oauth) imported from praw.ini
    reddit = praw.Reddit(site_name="wsb", requestor_class=LoggingRequestor,
                         comment_kind=COMMENT_PREFIX, redditor_kind=USER_PREFIX,
                         submission_kind=SUBMISSION_PREFIX)

    r_wsb = reddit.subreddit("wallstreetbets")

    # Fetch and save comment data
    wsb_data = get_comment_data(r_wsb)
    wsb_data.to_csv(f"{OUT_FOLDER}/wsb-data-{datetime.now()}.csv")

    # Calculate and save user sentiment data
    user_sentiment_data = xxx


def get_comment_data(subreddit: praw.models.Subreddit) -> pd.DataFrame:
    subreddit_data = pd.DataFrame()
    try:
        # Create a submission iterator and get submissions from the past two months.
        # From those, extract the comments and put them into a dataframe
        submissions_iterator = subreddit.stream().submissions()
        for submission in submissions_iterator:
            sub_date = datetime.utcfromtimestamp(submission.created_utc)

            # TODO first test with 2 days ago to see if i'm doing stuff right
            THRESHOLD_DATE = datetime.today() - timedelta(days=2)

            if sub_date > THRESHOLD_DATE:
                sub_data = fetch_data_from_submission(submission)
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


# TODO refactoring in due funzioni, una per fetchare dati da submission e costruire network,
# una che raggruppa commenti per autore presenti nel submission df e calcola sentiment


def fetch_data_from_submission(submission: praw.models.Submission) -> pd.DataFrame:
    """
    Fetch all the comments from a given submission and return a dataframe containing said comments.
    Title + selftext + author of the provided submission are treated as a comment in response to no user.

    :param submission:
    :return: dataframe containing the following columns: author, comment, responding_to
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
    print(f"(Fetching date interval: {TWO_MONTHS_AGO} - {date.today()})")

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

    df_data = {AUTHOR_COL: authors, COMMENT_COL: comments,
               RESPONDING_TO_COL: [comment_authors[p_id] for p_id in parent_ids]}
    return pd.DataFrame(data=df_data)


if __name__ == "__main__":
    main()
