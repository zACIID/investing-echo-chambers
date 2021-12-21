import time
import pandas as pd
from datetime import datetime, date, timedelta
from interactions import Interaction, SubredditInteractions
from sentiment import get_user_sentiment_df
from constants import USER_COL, TEXT_COL, INTERACTED_WITH_COL, SENTIMENT_COL


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
