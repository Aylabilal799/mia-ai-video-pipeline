"""schedule_utils.py -- parses a date/time typed in Discord into the RFC3339
UTC timestamp YouTube's publishAt field requires.
"""

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# The timezone your date/time in !miaschedule is interpreted in. Defaults to
# Pakistan time since that's where this bot is operated from; change in .env
# if that's wrong, or set to "UTC" to always type times in UTC.
SCHEDULE_TIMEZONE = os.getenv("SCHEDULE_TIMEZONE", "Asia/Karachi")

# YouTube needs some lead time between "scheduled" and "now" -- also gives
# generation itself time to finish before the publish moment arrives.
MIN_LEAD_MINUTES = int(os.getenv("SCHEDULE_MIN_LEAD_MINUTES", "10"))


class ScheduleParseError(Exception):
    pass


def parse_schedule_datetime(date_str: str, time_str: str) -> str:
    """date_str: 'YYYY-MM-DD', time_str: 'HH:MM' (24h), both interpreted in
    SCHEDULE_TIMEZONE. Returns an RFC3339 UTC string like
    '2026-08-20T09:30:00Z' for YouTube's publishAt, or raises
    ScheduleParseError with a message that's safe to show the user."""
    try:
        naive = datetime.strptime(date_str + " " + time_str, "%Y-%m-%d %H:%M")
    except ValueError:
        raise ScheduleParseError(
            "Couldn't parse that date/time. Use: YYYY-MM-DD HH:MM (24h), "
            "e.g. 2026-08-20 14:30."
        )

    try:
        local_dt = naive.replace(tzinfo=ZoneInfo(SCHEDULE_TIMEZONE))
    except Exception:
        raise ScheduleParseError(
            "Server is misconfigured: invalid SCHEDULE_TIMEZONE '" + SCHEDULE_TIMEZONE + "'."
        )

    now_utc = datetime.now(timezone.utc)
    delta_minutes = (local_dt.astimezone(timezone.utc) - now_utc).total_seconds() / 60

    if delta_minutes < MIN_LEAD_MINUTES:
        raise ScheduleParseError(
            "That time is too soon (or in the past). Pick a time at least "
            + str(MIN_LEAD_MINUTES) + " minutes from now, in " + SCHEDULE_TIMEZONE + " time."
        )

    utc_dt = local_dt.astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
