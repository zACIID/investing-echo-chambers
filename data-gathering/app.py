import glob

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from src.interactions import SubmissionsInteractionFetcher, get_interaction_df
from src.sentiment import get_user_sentiment_df, get_text_sentiment_df
from src.constants import USER_COL, TEXT_COL, INTERACTED_WITH_COL, SENTIMENT_COL

# last two weeks of december
ENDING_DATE = datetime(year=2021, month=12, day=31)
STARTING_DATE = datetime(year=2021, month=12, day=17)
DAYS_INTERVAL = (ENDING_DATE - STARTING_DATE).days

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

    for day in range(1, DAYS_INTERVAL + 1):
        day_info = f"[Day {day} of {DAYS_INTERVAL}]"

        # Initialization
        print(f"--------- {day_info} Initialization... ---------")
        from_date = STARTING_DATE + timedelta(days=day - 1)
        to_date = STARTING_DATE + timedelta(days=day)

        search_params = {
            "q": "(stocks)|(markets)|(stock market)|(investing)|(investment)",
            "after": int(from_date.timestamp()),
            "before": int(to_date.timestamp()),
            "limit": None
        }
        fetcher = SubmissionsInteractionFetcher(ini_site_name="project",
                                                replace_more_minimum=35,
                                                logger=print, **search_params)

        # Fetch and save comment data
        print(f"--------- {day_info} Fetching interactions... ---------")
        wsb_interactions = fetcher.fetch_interactions()
        print(f"--------- {day_info} interactions fetched ---------")

        # Create and save interactions dataframe
        interactions_df = get_interaction_df(interactions=wsb_interactions,
                                             user_out_col=USER_COL,
                                             text_out_col=TEXT_COL,
                                             interacted_with_out_col=INTERACTED_WITH_COL)
        if len(interactions_df) > 0:
            print(f"--------- {day_info} Saving interactions into csv... ---------")
            interactions_df.to_csv(
                f"{INTERACTIONS_DAY_TO_DAY_FOLDER}/interactions__{from_date.date()}_{to_date.date()}.csv",
                index=False
            )

        # Calculate and save text sentiment data
        print(f"--------- {day_info} Calculating text sentiment... ---------")
        text_sentiment_df = get_text_sentiment_df(interactions=wsb_interactions,
                                                  text_out_col=TEXT_COL,
                                                  sentiment_out_col=SENTIMENT_COL)
        if len(text_sentiment_df) > 0:
            print(f"--------- {day_info} Saving text sentiment into csv... ---------")
            text_sentiment_df.to_csv(
                f"{TEXT_SENTIMENT_DAY_TO_DAY_FOLDER}/text-sentiment__{from_date.date()}_{to_date.date()}.csv",
                index=False
            )

        # Calculate and save user sentiment data
        print(f"--------- {day_info} Calculating users' sentiment... ---------")
        user_sentiment_df = get_user_sentiment_df(interactions=wsb_interactions,
                                                  user_out_col=USER_COL,
                                                  sentiment_out_col=SENTIMENT_COL)
        if len(user_sentiment_df) > 0:
            print(f"--------- {day_info} Saving users' sentiment into csv... ---------")
            user_sentiment_df.to_csv(
                f"{USER_SENTIMENT_DAY_TO_DAY_FOLDER}/user-sentiment__{from_date.date()}_{to_date.date()}.csv",
                index=False
            )

        print(f"--------- {day_info} Completed ---------")

    print("---------------- creating final interaction dataset ----------------")
    interactions_final = concat_stored_datasets(f"{INTERACTIONS_DAY_TO_DAY_FOLDER}/interactions__*.csv")
    interactions_final.to_csv(f"{OUT_FOLDER}/interactions.csv", index=False)

    print("---------------- creating final text sentiment dataset ----------------")
    text_sentiment_final = concat_stored_datasets(f"{TEXT_SENTIMENT_DAY_TO_DAY_FOLDER}/text-sentiment__*.csv")
    text_sentiment_final.to_csv(f"{OUT_FOLDER}/text-sentiment.csv", index=False)

    print("---------------- creating final user sentiment dataset ----------------")
    user_sentiment_final = concat_stored_datasets(f"{USER_SENTIMENT_DAY_TO_DAY_FOLDER}/user-sentiment__*.csv")

    # Recalculate average user sentiment
    user_sentiment_final = user_sentiment_final.groupby(by=[USER_COL], as_index=False).mean()
    user_sentiment_final.to_csv(f"{OUT_FOLDER}/user-sentiment.csv", index=False)

    print("__________________________________________________________________________________________________________")
    print(f"Data for interval [{STARTING_DATE} - {ENDING_DATE}] successfully fetched.")


def setup_directories():
    # Create if not exist
    Path(OUT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(INTERACTIONS_DAY_TO_DAY_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(TEXT_SENTIMENT_DAY_TO_DAY_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(USER_SENTIMENT_DAY_TO_DAY_FOLDER).mkdir(parents=True, exist_ok=True)


def concat_stored_datasets(file_path_pattern: str) -> pd.DataFrame:
    """
    :param file_path_pattern: append only datasets whose path matches this pattern
    :return: single dataframe containing all the datasets that matched the provided pattern
    """
    # for each file that matches pattern,
    # read and concat into a single dataframe
    path_list = glob.glob(f"{file_path_pattern}")
    df = pd.DataFrame()
    for path in path_list:
        to_append = pd.read_csv(path)
        df = df.append(to_append)

    return df


if __name__ == "__main__":
    main()
