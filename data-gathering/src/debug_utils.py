import glob

import pandas as pd

from src.interactions import get_interactions_from_df
from src.sentiment import get_user_sentiment_df, get_text_sentiment_df
from src.constants import USER_COL, TEXT_COL, INTERACTED_WITH_COL, SENTIMENT_COL

# Output directories
OUT_FOLDER = "./output"
INTERACTIONS_DAY_TO_DAY_FOLDER = f"{OUT_FOLDER}/interactions_day-to-day"
TEXT_SENTIMENT_DAY_TO_DAY_FOLDER = f"{OUT_FOLDER}/text-sentiment_day-to-day"
USER_SENTIMENT_DAY_TO_DAY_FOLDER = f"{OUT_FOLDER}/user-sentiment_day-to-day"


def test_main():
    int_df = pd.read_csv(f"{OUT_FOLDER}/wsb-interactions.csv")
    interactions = get_interactions_from_df(int_df, USER_COL, TEXT_COL, INTERACTED_WITH_COL)
    user_sent_df = get_user_sentiment_df(interactions, USER_COL, SENTIMENT_COL)
    print(len(user_sent_df))
    user_sent_df.to_csv(f"{OUT_FOLDER}/wsb-user-sentiment.csv", index=False)


def day_to_day_text_from_interactions():
    # Used because the code for text-sentiment datasets was added after having fetched the data
    print("---------------- creating text sentiment day-to-day datasets ----------------")
    import re
    path_list = glob.glob(f"{INTERACTIONS_DAY_TO_DAY_FOLDER}/wsb-interactions__*.csv")
    for path in path_list:
        int_df = pd.read_csv(path)
        interactions = get_interactions_from_df(interactions_df=int_df,
                                                user_col=USER_COL,
                                                text_col=TEXT_COL,
                                                interacted_with_col=INTERACTED_WITH_COL)

        wsb_text_sentiment_df = get_text_sentiment_df(interactions,
                                                      text_out_col=TEXT_COL,
                                                      sentiment_out_col=SENTIMENT_COL)

        file_date = re.match(".*__(.*).csv", path).group(1)
        wsb_text_sentiment_df.to_csv(f"{TEXT_SENTIMENT_DAY_TO_DAY_FOLDER}/wsb-text-sentiment__{file_date}.csv")
