import time
from typing import Union
from abc import ABC
from collections import Generator
from praw import Reddit
from praw.models import Submission, Comment
from prawcore import Requestor
from psaw import PushshiftAPI
from datetime import datetime
from src.constants import COMMENT_PREFIX, USER_PREFIX, SUBMISSION_PREFIX


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


class Interaction(ABC):
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
        author = get_author_username_safe(submission)

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
        author = get_author_username_safe(comment)
        interacted_with = get_author_username_safe(parent)

        text_data = comment.body

        super(CommentInteraction, self).__init__(author, text_data, interacted_with)


def extract_id_safe(praw_obj: Union[Comment, Submission]) -> str:
    """
    Used to extract the actual id, net of prefixes, from a given praw object.
    :param praw_obj: praw object representing either comment or submission
    :return: id, net of reddit api prefixes, or "N/A" if the object has no id
    """

    # Some objects yielded by PRAW do not have the id field
    obj_id = getattr(praw_obj, "id", "N/A")

    # Remove all the prefixes so that only the actual id remains
    return remove_kind_prefixes(obj_id)


def extract_parent_id_safe(praw_obj: Union[Comment, Submission]) -> str:
    """
    Used to extract the parent id, net of prefixes, from a given praw object.
    :param praw_obj: praw object representing either comment or submission
    :return: parent id, net of reddit api prefixes,
        or "N/A" if the object has no parent id
    """

    # Some objects yielded by PRAW do not have the id field
    obj_id = getattr(praw_obj, "parent_id", "N/A")

    # Remove all the prefixes so that only the actual id remains
    return remove_kind_prefixes(obj_id)


def remove_kind_prefixes(obj_id: str) -> str:
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


def get_author_username_safe(praw_obj: Union[Comment, Submission]) -> str:
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


class SubredditInteractions(object):
    """
    Class used to fetch the necessary data to build a user network of
    a certain subreddit in a specified time interval.
    Such network is made of interactions between users,
    which essentially are comments/submissions and relative responses.
    """
    def __init__(self, subreddit: str, ini_site_name: str = "DEFAULT",
                 date_after: datetime = None, date_before: datetime = None,
                 logger: callable(str) = None):
        """
        :param subreddit: name of the subreddit to fetch the data from
        :param ini_site_name: name of the site in the praw.ini file placed in the current working directory.
            Used to access the reddit api via a praw.Reddit instance.
        :param date_after: date after which data is fetched (included).
            Defaults to the current day if not specified.
        :param date_before: date before which data is fetched (included).
            Defaults to the current day if not specified.
        :param logger: function that performs logging operations.
            If nothing is specified, nothing is logged.
        """

        self._reddit = Reddit(site_name=ini_site_name, requestor_class=LoggingRequestor)
        self._psaw = PushshiftAPI(r=self._reddit)
        self._subreddit = self._reddit.subreddit(subreddit)

        self._date_before = date_before if date_before is not None else datetime.today()
        self._date_after = date_after if date_after is not None else datetime.today()

        self._logger = logger

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

            sub_data = self._fetch_data_from_submission(submission)
            fetched_data += sub_data

        return fetched_data

    def _get_submission_generator(self) -> Generator[Submission]:
        """
        Returns the ids of all the submissions in this instance specified time interval
        :return:
        """
        # Reference tutorial: https://towardsdatascience.com/how-to-collect-a-reddit-dataset-c369de539114

        # Query PSAW to get all the ids in the specified time interval
        # PSAW is needed because PRAW goes at maximum 1000 elements back in time,
        # while PSAW consults the pushshift.io database, which stores historical reddit data
        ts_after = int(self._date_after.timestamp())
        ts_before = int(self._date_before.timestamp())
        psaw_sub_generator = self._psaw.search_submissions(
            after=ts_after,
            before=ts_before,
            filter=['id'],
            subreddit=self._subreddit,
            limit=None  # get all the submissions in the specified timeframe
        )

        # For each PSAW id, get the PRAW object for the submission,
        # which is easier to handle than PSAW raw data
        for psaw_sub in psaw_sub_generator:
            submission_id = psaw_sub.id

            # Skip submission if outside date range
            # it can happen because they are fetched in batches
            sub_date = datetime.utcfromtimestamp(psaw_sub.created_utc)
            if sub_date <= self._date_after or sub_date >= self._date_before:
                self._log_message("Skipping submission...")
                continue

            yield self._reddit.submission(id=submission_id)

    def _fetch_data_from_submission(self, submission: Submission) -> list[Interaction]:
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

        # Log some submission info to keep track of progress
        unix_time_date = submission.created_utc
        creation_date = datetime.utcfromtimestamp(unix_time_date)
        self._log_message(f"Fetching submission '{submission.title}', created at: {creation_date}")
        self._log_message(f"(Date interval: {self._date_after} - {self._date_before})")

        # The current submission is parent of its top-level comments
        sub_id = extract_id_safe(submission)
        ids_to_objects[sub_id] = submission

        # Using replace_more is like pressing an all the buttons that say "load more comments"
        # With the limit set to None, it tells praw to basically press all the "load more" buttons
        # that load at least _threshold_ comments
        # (30 is arbitrary, but under that it seems like a waste of a request)
        # See https://praw.readthedocs.io/en/latest/tutorials/comments.html for more info
        submission.comments.replace_more(limit=None, threshold=30)

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
            comment_id = extract_id_safe(c)
            ids_to_objects[comment_id] = c

        # Retrieve the parent of the fetched comments and create the interactions
        for c in fetched_comments:
            parent_id = extract_parent_id_safe(c)
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
