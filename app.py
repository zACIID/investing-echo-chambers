import glob
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from src.interactions import SubredditInteractions, get_interaction_df, Interaction
from src.sentiment import get_user_sentiment_df, get_text_sentiment_df
from src.constants import USER_COL, TEXT_COL, INTERACTED_WITH_COL, SENTIMENT_COL


# Get data from up to two months ago
DAYS_INTERVAL = 60
STARTING_DATE = datetime.today() - timedelta(days=DAYS_INTERVAL)

# Output directories
OUT_FOLDER = "./output"
INTERACTIONS_DAY_TO_DAY_FOLDER = f"{OUT_FOLDER}/interactions_day-to-day"
TEXT_SENTIMENT_DAY_TO_DAY_FOLDER = f"{OUT_FOLDER}/text-sentiment_day-to-day"
USER_SENTIMENT_DAY_TO_DAY_FOLDER = f"{OUT_FOLDER}/user-sentiment_day-to-day"


def main():
    # To keep memory consumption low and possibly avoid losing
    # a lot of data, divide the task in days and
    # dump data into csv file after each day
    setup_directories()

    for day in range(1, DAYS_INTERVAL+1):
        day_info = f"[Day {day}]"

        # Initialization
        print(f"--------- {day_info} Initializing r/wsb... ---------")
        from_date = STARTING_DATE + timedelta(days=day-1)
        to_date = STARTING_DATE + timedelta(days=day)
        wsb = SubredditInteractions(subreddit="wallstreetbets", ini_site_name="wsb",
                                    date_after=from_date, date_before=to_date,
                                    logger=print)

        # Fetch and save comment data
        print(f"--------- {day_info} Fetching r/wsb interactions... ---------")
        wsb_interactions = wsb.fetch_interactions()
        print(f"--------- {day_info} r/wsb interactions fetched ---------")

        # Create and save interactions dataframe
        interactions_df = get_interaction_df(interactions=wsb_interactions,
                                             user_out_col=USER_COL,
                                             text_out_col=TEXT_COL,
                                             interacted_with_out_col=INTERACTED_WITH_COL)
        if len(interactions_df) > 0:
            print(f"--------- {day_info} Saving r/wsb interactions into csv... ---------")
            interactions_df.to_csv(
                f"{INTERACTIONS_DAY_TO_DAY_FOLDER}/wsb-interactions__{from_date.date()}_{to_date.date()}.csv")

        # Calculate and save text sentiment data
        print(f"--------- {day_info} Calculating r/wsb text sentiment... ---------")
        text_sentiment_df = get_text_sentiment_df(interactions=wsb_interactions,
                                                  text_out_col=TEXT_COL,
                                                  sentiment_out_col=SENTIMENT_COL)
        if len(text_sentiment_df) > 0:
            print(f"--------- {day_info} Saving r/wsb text sentiment into csv... ---------")
            text_sentiment_df.to_csv(
                f"{TEXT_SENTIMENT_DAY_TO_DAY_FOLDER}/wsb-text-sentiment__{from_date.date()}_{to_date.date()}.csv")

        # Calculate and save user sentiment data
        print(f"--------- {day_info} Calculating r/wsb users' sentiment... ---------")
        user_sentiment_df = get_user_sentiment_df(interactions=wsb_interactions,
                                                  user_out_col=USER_COL,
                                                  sentiment_out_col=SENTIMENT_COL)
        if len(user_sentiment_df) > 0:
            print(f"--------- {day_info} Saving r/wsb users' sentiment into csv... ---------")
            user_sentiment_df.to_csv(
                f"{USER_SENTIMENT_DAY_TO_DAY_FOLDER}/wsb-user-sentiment__{from_date.date()}_{to_date.date()}.csv")

        print(f"--------- {day_info} Completed ---------")

        print("---------------- creating final interaction dataset ----------------")
        interactions_final = append_stored_datasets(f"{INTERACTIONS_DAY_TO_DAY_FOLDER}/wsb-interactions__*.csv")
        interactions_final.to_csv(f"{OUT_FOLDER}/wsb-interactions.csv", index=False)

        print("---------------- creating final text sentiment dataset ----------------")
        text_sentiment_final = append_stored_datasets(f"{TEXT_SENTIMENT_DAY_TO_DAY_FOLDER}/wsb-text-sentiment__*.csv")
        text_sentiment_final.to_csv(f"{OUT_FOLDER}/wsb-text-sentiment.csv", index=False)

        print("---------------- creating final user sentiment dataset ----------------")
        user_sentiment_final = append_stored_datasets(f"{USER_SENTIMENT_DAY_TO_DAY_FOLDER}/wsb-user-sentiment__*.csv")
        user_sentiment_final = user_sentiment_final.groupby(by=[USER_COL]).mean()
        user_sentiment_final.to_csv(f"{OUT_FOLDER}/wsb-user-sentiment.csv", index=False)

    print("__________________________________________________________________________________________________________")
    print(f"Data for interval {STARTING_DATE} - {STARTING_DATE + timedelta(days=DAYS_INTERVAL)} successfully fetched.")


def setup_directories():
    # Create if not exist
    Path(OUT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(INTERACTIONS_DAY_TO_DAY_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(TEXT_SENTIMENT_DAY_TO_DAY_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(USER_SENTIMENT_DAY_TO_DAY_FOLDER).mkdir(parents=True, exist_ok=True)


def append_stored_datasets(file_path_pattern: str) -> pd.DataFrame:
    """
    :param file_path_pattern: append only datasets whose path matches this pattern
    """

    # for file that matches pattern, append and then save into one csv file
    # TODO implement this, then call this in main at the end for every type of file
    #   for user-sentiment, it needs to be grouped by + mean() before being stored again
    #   maybe create "partial" folder where all the day-to-day data is stored
    path_list = glob.glob(f"{file_path_pattern}")
    df = pd.DataFrame()
    for path in path_list:
        to_append = pd.read_csv(path)
        df = df.append(to_append)

    return df


if __name__ == "__main__":
    main()


def day_to_day_text_from_interactions():
    # Used because the code for text-sentiment datasets was added after having fetched the data
    print("---------------- creating text sentiment day-to-day datasets ----------------")
    import re
    path_list = glob.glob(f"{INTERACTIONS_DAY_TO_DAY_FOLDER}/wsb-interactions__*.csv")
    for path in path_list:
        wsb_int = pd.read_csv(path)

        interactions = []
        for row in wsb_int.itertuples():
            user = getattr(row, USER_COL)
            text = getattr(row, TEXT_COL)
            interacted_with = getattr(row, INTERACTED_WITH_COL)
            temp = Interaction(user, text, interacted_with)
            interactions.append(temp)

        wsb_text_sentiment_df = get_text_sentiment_df(interactions,
                                                      text_out_col=TEXT_COL,
                                                      sentiment_out_col=SENTIMENT_COL)
        file_date = re.match(".*__(.*).csv", path).group(1)
        wsb_text_sentiment_df.to_csv(f"{TEXT_SENTIMENT_DAY_TO_DAY_FOLDER}/wsb-text-sentiment__{file_date}.csv")
