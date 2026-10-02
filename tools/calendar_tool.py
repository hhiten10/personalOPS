"""
Google Calendar read-only integration for personalOPS.

Handles the OAuth token lifecycle (first-run browser auth, then cached
refresh) and exposes a single LangChain tool: get_upcoming_events().
"""

import os
import datetime
from langchain_core.tools import tool

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Read-only scope — this app can never create/edit/delete events, only list them.
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# Resolve paths relative to this file so it works regardless of cwd
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(_BASE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(_BASE_DIR, "token.json")


def _get_credentials() -> Credentials:
    """
    Loads cached credentials from token.json if present and valid.
    If expired, refreshes silently using the refresh token.
    If no token exists at all, runs the one-time browser OAuth flow
    and caches the result so this only ever happens once.
    """
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH, SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Cache for next time — this is what avoids re-authenticating every run
        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())

    return creds


def _fetch_upcoming_events(max_results: int = 10) -> list[dict]:
    """Low-level fetch, kept separate from the @tool wrapper for testability."""
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)

    now = datetime.datetime.utcnow().isoformat() + "Z"  # 'Z' = UTC

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    return events_result.get("items", [])


@tool
def get_upcoming_events(max_results: int = 10) -> str:
    """
    Fetch the user's upcoming Google Calendar events (read-only).

    Use this when the user asks about their schedule, free time, upcoming
    commitments, or when planning study/gym sessions that need to account
    for existing calendar events.

    Args:
        max_results: Maximum number of upcoming events to return (default 10).

    Returns:
        A human-readable summary of upcoming events with their start times.
    """
    events = _fetch_upcoming_events(max_results=max_results)

    if not events:
        return "No upcoming events found on the calendar."

    lines = []
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        summary = event.get("summary", "(no title)")
        lines.append(f"- {summary} at {start}")

    return "Upcoming events:\n" + "\n".join(lines)