"""youtube_auth_setup.py — run this ONCE, manually, on the VPS to authorize
the Google account that will own the uploaded videos.
"""

import os
from google_auth_oauthlib.flow import InstalledAppFlow

from youtube_auth import YOUTUBE_SCOPES, CLIENT_SECRETS_FILE, TOKEN_FILE

OAUTH_PORT = int(os.getenv("YOUTUBE_OAUTH_PORT", "8080"))


def main():
    if not os.path.exists(CLIENT_SECRETS_FILE):
        raise SystemExit(
            f"Missing {CLIENT_SECRETS_FILE}. Download your OAuth Client ID JSON "
            "from Google Cloud Console and save it there first."
        )

    print(
        "\nBefore continuing, open a NEW terminal on your LOCAL machine and run:\n\n"
        f"    ssh -L {OAUTH_PORT}:localhost:{OAUTH_PORT} <your-user>@<this-vps-host>\n\n"
        "Leave that connected, then come back here.\n"
    )
    input("Press Enter once the SSH tunnel is up... ")

    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, YOUTUBE_SCOPES)

    creds = flow.run_local_server(
        host="localhost",
        port=OAUTH_PORT,
        open_browser=False,
        authorization_prompt_message=(
            "\nOn your LOCAL machine's browser (through the SSH tunnel), open:\n\n{url}\n"
        ),
        success_message="Authorization complete -- you can close this browser tab now.",
    )

    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())

    print(f"\nSaved YouTube OAuth token to {TOKEN_FILE}. Uploads will now work.")


if __name__ == "__main__":
    main()
