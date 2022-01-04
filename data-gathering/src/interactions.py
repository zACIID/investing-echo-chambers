import time
import pandas as pd
from typing import Union
from collections import Generator
from praw import Reddit
from praw.models import Submission, Comment
from prawcore import Requestor
from psaw import PushshiftAPI
from datetime import datetime
from constants import COMMENT_PREFIX, USER_PREFIX, SUBMISSION_PREFIX


class LoggingRequestor(Requestor):
    prev_request_time = 0

    def request(self, *args, timeout=None, **kwargs):
        """
        Wrap the request method with logging capabilities.
        """
        response = super().request(*args, **kwargs)

        print(f"Response from: {response.url}")

        curr_time = time.perf_counter()
        elapsed_time = curr_time - self.prev_request_time
        print(f"Time from previous request: {round(elapsed_time, 3)}s")
        self.prev_request_time = curr_time

        return response


class Interaction(object):
    """
    Abstract class representing an interaction between two users
    """
    def __init__(self, user: str, text_data: str, interacted_with: str):
        self.user = user
        self.text_data = text_data
        self.interacted_with = interacted_with


class SubmissionInteraction(Interaction):
    """
    Class that represents the network data gathered from a reddit submission.
    Since a submission is at the root of all the comments, it isn't really in
    response to anyone, so it is treated as if the author is responding to itself.
    (author == responding_to)
    Also, text_data is made of both the title and the submission text.
    """
    def __init__(self, submission: Submission):
        # Convert the author to string to get the username
        # without activating the lazy Redditor instance and causing
        # an api call
        author = _get_author_username_safe(submission)

        text_data = f"{submission.title} - {submission.selftext}"

        super(SubmissionInteraction, self).__init__(author, text_data, author)


class CommentInteraction(Interaction):
    """
    Class that represents the network data gathered from a reddit submission
    """
    def __init__(self, comment: Comment, parent: Union[Comment, Submission]):
        # Convert the author to string to get the username
        # without activating the lazy Redditor instance and causing
        # an api call
        author = _get_author_username_safe(comment)
        interacted_with = _get_author_username_safe(parent)

        text_data = comment.body

        super(CommentInteraction, self).__init__(author, text_data, interacted_with)


def _extract_id_safe(praw_obj: Union[Comment, Submission]) -> str:
    """
    Used to extract the actual id, net of prefixes, from a given praw object.
    :param praw_obj: praw object representing either comment or submission
    :return: id, net of reddit api prefixes, or "N/A" if the object has no id
    """

    # Some objects yielded by PRAW do not have the id field
    obj_id = getattr(praw_obj, "id", "N/A")

    # Remove all the prefixes so that only the actual id remains
    return _remove_kind_prefixes(obj_id)


def _extract_parent_id_safe(praw_obj: Union[Comment, Submission]) -> str:
    """
    Used to extract the parent id, net of prefixes, from a given praw object.
    :param praw_obj: praw object representing either comment or submission
    :return: parent id, net of reddit api prefixes,
        or "N/A" if the object has no parent id
    """

    # Some objects yielded by PRAW do not have the id field
    obj_id = getattr(praw_obj, "parent_id", "N/A")

    # Remove all the prefixes so that only the actual id remains
    return _remove_kind_prefixes(obj_id)


def _remove_kind_prefixes(obj_id: str) -> str:
    """
    Removes the reddit api prefixes such as t1, t2, etc.
    from the given id string
    :param obj_id:
    :return: id string, net of prefixes
    """
    actual_id = obj_id.replace(COMMENT_PREFIX, "")
    actual_id = actual_id.replace(USER_PREFIX, "")
    actual_id = actual_id.replace(SUBMISSION_PREFIX, "")

    return actual_id


def _get_author_username_safe(praw_obj: Union[Comment, Submission]) -> str:
    """
    Function that handles the case where the author of something (comment, submission, etc.)
    has been deleted, in which case it is represented as None by PRAW.
    :param praw_obj: praw object representing either comment or submission
    :return: author's username or "N/A"
    """
    if praw_obj.author is None:
        return "N/A"
    else:
        return f"{praw_obj.author}"


class SubmissionsInteractionFetcher(object):
    """
    Class used to fetch the necessary data to build a user network from certain
    reddit submissions (posts) in a specified time interval.
    Such network is made of interactions between users,
    which essentially are comments/submissions and relative responses.
    """
    def __init__(self, ini_site_name: str = "DEFAULT",
                 replace_more_minimum: int = 30,
                 logger: callable(str) = None, **search_params):
        """
        :param ini_site_name: name of the site in the praw.ini file placed in the current working directory.
            Used to access the reddit api via a praw.Reddit instance.
        :param replace_more_minimum: minimum comments that a PRAW replace_more request must
            fetch in order to be sent. See the following link for more info:
            https://praw.readthedocs.io/en/latest/tutorials/comments.html#the-replace-more-method
        :param logger: function that performs logging operations.
            If nothing is specified, nothing is logged.
        :param search_params: search parameters for submissions.
            See the following links to get more info on what parameters are accepted:
            https://github.com/dmarx/psaw
            https://github.com/pushshift/api#searching-submissions

            Additional examples:
            https://melaniewalsh.github.io/Intro-Cultural-Analytics/04-Data-Collection/14-Reddit-Data.html
            https://towardsdatascience.com/how-to-collect-a-reddit-dataset-c369de539114
        """
        self._reddit = Reddit(site_name=ini_site_name, requestor_class=LoggingRequestor)
        self._psaw = PushshiftAPI(r=self._reddit)

        self._replace_more_minimum = replace_more_minimum
        self._logger = logger
        self._search_params = search_params

    def fetch_interactions(self) -> list[Interaction]:
        """
        Fetches the network data for the subreddit specified when creating this instance.
        :return: list of network data
        """
        fetched_data: list[Interaction] = []
        fetched_counter = 0
        submission_iterator = self._get_submission_generator()
        for submission in submission_iterator:
            # Log progress
            fetched_counter += 1
            self._log_message(f"[Submission #{fetched_counter}]")

            sub_data = self._fetch_interactions_from_submission(submission)
            fetched_data += sub_data

        return fetched_data

    def _get_submission_generator(self) -> Generator[Submission]:
        """
        Returns a generator of submissions based on this instance's search parameters.
        :return: submission generator
        """

        # Query PSAW to get all the submission ids corresponding to the
        # provided search parameters.
        # PSAW is needed because PRAW submission generator
        # goes at maximum 1000 elements back in time,
        # while PSAW consults the pushshift.io database,
        # which stores historical reddit data.
        # A "filter" search param is added (or replaced) because only
        # the id field of the submission is needed, the rest of the data
        # is fetched using PRAW.
        self._search_params["filter"] = ["id"]
        psaw_sub_generator = self._psaw.search_submissions(**self._search_params)

        # For each PSAW id, get the PRAW object for the submission,
        # which is easier to handle than PSAW raw data
        for psaw_sub in psaw_sub_generator:
            submission_id = psaw_sub.id

            # Log some submission info to keep track of progress
            submission = self._reddit.submission(id=submission_id)
            unix_time_date = submission.created_utc
            creation_date = datetime.utcfromtimestamp(unix_time_date)
            self._log_message(f"Fetching submission '{submission.title}', created at: {creation_date}")

            yield submission

    def _fetch_interactions_from_submission(self, submission: Submission) -> list[Interaction]:
        """
        Fetches the network interaction data from the given submission
        :param submission: reddit submission to extract interactions from
        :return:
        """
        interactions: list[Interaction] = []

        sub_interaction = SubmissionInteraction(submission)
        interactions.append(sub_interaction)

        # Pair the comment/submission id with the actual fetched object
        # used to resolve comment.parent_id
        ids_to_objects = {}

        # The current submission is parent of its top-level comments
        sub_id = _extract_id_safe(submission)
        ids_to_objects[sub_id] = submission

        # Using replace_more is like pressing an all the buttons that say "load more comments"
        # With the limit set to None, it tells praw to basically press all the "load more" buttons
        # that load at least _threshold_ comments
        # (30 is arbitrary, but under that it seems like a waste of a request)
        # See https://praw.readthedocs.io/en/latest/tutorials/comments.html for more info
        submission.comments.replace_more(limit=None, threshold=self._replace_more_minimum)

        # tot_comments might not be exact because it also counts deleted comments, blocked comments, etc
        tot_comments = submission.num_comments

        fetched_counter = 0
        fetched_comments: list[Comment] = []

        comments_iterator = submission.comments.list()
        for c in comments_iterator:
            # Progress logging
            fetched_counter += 1
            progress_pct = round(((fetched_counter / tot_comments) * 100), 2)
            self._log_message(f"Fetching comment #{fetched_counter} of max {tot_comments} [{progress_pct}%]")

            fetched_comments.append(c)

            # Build the dictionary later used to go from parent_id to parent object
            comment_id = _extract_id_safe(c)
            ids_to_objects[comment_id] = c

        # Retrieve the parent of the fetched comments and create the interactions
        for c in fetched_comments:
            parent_id = _extract_parent_id_safe(c)
            parent = ids_to_objects[parent_id]

            comm_interaction = CommentInteraction(c, parent)
            interactions.append(comm_interaction)

        return interactions

    def _log_message(self, msg: str):
        """
        Logs a message using the logger provided to this instance, if any.
        :param msg: message to log
        """
        if self._logger is not None:
            self._logger(msg)


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


def get_interactions_from_df(interactions_df: pd.DataFrame,
                             user_col: str, text_col: str,
                             interacted_with_col: str) -> list[Interaction]:
    """
    Get a list of interaction objects from an interaction dataframe.
    :param interactions_df: interactions dataframe
    :param user_col: name of the user column (first) in the provided dataframe
    :param text_col: name of the text column (second)  in the provided dataframe
    :param interacted_with_col: name of the interacted_with column (third)
        in the provided dataframe
    :return: interaction objects list
    """
    interactions = []
    for row in interactions_df.itertuples():
        user = getattr(row, user_col)
        text = getattr(row, text_col)
        interacted_with = getattr(row, interacted_with_col)
        temp = Interaction(user, text, interacted_with)
        interactions.append(temp)

    return interactions