import praw
import praw.models
import pandas as pd
from prawcore import Requestor


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


def main():
    # some settings (including oauth) imported from praw.ini
    reddit = praw.Reddit(site_name="wsb", requestor_class=LoggingRequestor,
                         comment_kind=COMMENT_PREFIX, redditor_kind=USER_PREFIX,
                         submission_kind=SUBMISSION_PREFIX)

    r_wsb = reddit.subreddit("wallstreetbets")

    # Create a submission iterator and get submissions from the past two months.
    # From those, extract the comments and put them into a dataframe
    submission_iterator = r_wsb.new()
    comments = pd.DataFrame()
    for submission in submission_iterator:
        temp = fetch_comments_from_submission(submission)
        comments.append(temp)

    # TODO how to treat data from the submission itself (i.e. what to do with title and description?)



def fetch_comments_from_submission(submission: praw.models.Submission) -> pd.DataFrame:
    """
    Fetch all the comments from a given submission and return a dataframe containing said comments.

    :param submission:
    :return: dataframe containing all the comments
    """
    # TODO important info to put in the dataframe: author, comment, in_response_to
    # in_response_to identifies the author of the parent comment
    # (or submission, if the current comment is already top-level).
    # Needed to build the user network.
    pass


if __name__ == "__main__":
    main()