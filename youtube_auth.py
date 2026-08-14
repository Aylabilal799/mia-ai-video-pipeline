"""youtube_auth.py — OAuth credential loading for the YouTube Data API.

This module ONLY loads/refreshes an already-consented token. It does not run
the interactive consent flow itself -- that's a one-time manual step done via
youtube_auth_setup.py, run once on the VPS by a human (uploading requires a
real Google account's consent; it can't be automated headlessly).
"""

import os
import logging

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

logger = logging.getLogger(__name__)

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

CLIENT_SECRETS_FILE = os.getenv(
    "YOUTUBE_CLIENT_SECRETS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube_client_secret.json"),
)
TOKEN_FILE = os.getenv(
    "YOUTUBE_TOKEN_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube_token.json"),
)


def get_credentials() -> Credentials:
    """Loads the cached user token and refreshes it if expired. Raises with
    a clear message if youtube_auth_setup.py hasn't been run yet."""
    if not os.path.exists(TOKEN_FILE):
        raise RuntimeError(
            f"No YouTube OAuth token found at {TOKEN_FILE}. "
            "Run `python youtube_auth_setup.py` once on this VPS to authorize "
            "the uploader's Google account, then retry."
        )

    creds = Credentials.from_authorized_user_file(TOKEN_FILE, YOUTUBE_SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        logger.info("[YouTube Auth] Refreshed access token")

    if not creds or not creds.valid:
        raise RuntimeError(
            "YouTube OAuth token is invalid and could not be refreshed. "
            "Re-run `python youtube_auth_setup.py` on this VPS."
        )

    return creds
