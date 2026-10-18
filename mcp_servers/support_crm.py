"""MCP server: the support desk's users and tickets (Day 33).

The CRM of the support assistant is two JSON files under data/support -
synthetic users and tickets, no real people. This server is the only thing
that reads them, and it answers three narrow questions:

    get_ticket(ticket_id)        one ticket
    get_user(user_id)            the support-relevant part of one user
    get_user_tickets(user_id)    that user's tickets, as summaries

Deliberately narrow:

- identifiers are validated before anything is read (TICKET-<digits>,
  USER-<digits>); the files themselves are fixed, so no caller can name a
  path, and there is no list-everything tool;
- a user record carries only what support needs - language and plan - and
  nothing else is returned even if a file grows more fields;
- a missing file, broken JSON or a record without its required fields is an
  error with a reason the caller can act on, never an empty answer that
  looks like "no such ticket".

Errors are raised as tool errors whose message starts with a short kind:
``ticket_not_found``, ``user_not_found``, ``invalid_ticket_id``,
``invalid_user_id``, ``malformed_record``, ``support_data_unavailable``.

    python -m mcp_servers.support_crm
"""

import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

SERVER_NAME = "support-crm"
SERVER_VERSION = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: Where users.json and tickets.json live. The project's data directory by
#: default; SUPPORT_DATA_DIR points the server at another (the tests do).
DATA_DIR = Path(os.getenv("SUPPORT_DATA_DIR", str(PROJECT_ROOT / "data" / "support")))

TICKET_ID = re.compile(r"^TICKET-\d{1,8}$")
USER_ID = re.compile(r"^USER-\d{1,8}$")
MAX_TICKETS_PER_USER = 50


class SupportDataError(Exception):
    """The CRM could not answer. ``kind`` is the short reason a caller branches on."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


class Ticket(BaseModel):
    ticket_id: str
    user_id: str
    status: str
    category: str
    subject: str
    description: str = ""
    platform: str = ""
    app_version: str = ""
    previous_attempts: list[str] = []
    created_at: str = ""


class User(BaseModel):
    """What support may know about a user: the language to answer in and the plan."""

    user_id: str
    language: str = "ru"
    subscription: str = ""


class TicketSummary(BaseModel):
    ticket_id: str
    status: str
    category: str
    subject: str


class UserTickets(BaseModel):
    user_id: str
    tickets: list[TicketSummary]


def _records(name: str, key: str, data_dir: Path) -> list:
    path = data_dir / name

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SupportDataError("support_data_unavailable", f"{name} is missing") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupportDataError("support_data_unavailable", f"{name} is not readable JSON") from error

    records = document.get(key) if isinstance(document, dict) else document

    if not isinstance(records, list):
        raise SupportDataError("support_data_unavailable", f"{name} has no '{key}' list")

    return records


def _check(pattern: re.Pattern, value: str, kind: str) -> str:
    text = (value or "").strip().upper()

    if not pattern.match(text):
        raise SupportDataError(kind, f"{value!r} is not a valid identifier")

    return text


def find_ticket(ticket_id: str, data_dir: Path = DATA_DIR) -> Ticket:
    wanted = _check(TICKET_ID, ticket_id, "invalid_ticket_id")

    for record in _records("tickets.json", "tickets", data_dir):
        if isinstance(record, dict) and str(record.get("ticket_id", "")).upper() == wanted:
            try:
                return Ticket(**record)
            except ValidationError as error:
                raise SupportDataError("malformed_record", f"{wanted} is missing required fields") from error

    raise SupportDataError("ticket_not_found", f"there is no ticket {wanted}")


def find_user(user_id: str, data_dir: Path = DATA_DIR) -> User:
    wanted = _check(USER_ID, user_id, "invalid_user_id")

    for record in _records("users.json", "users", data_dir):
        if isinstance(record, dict) and str(record.get("user_id", "")).upper() == wanted:
            try:
                # Only the fields the model declares - whatever else the file
                # holds about a person stays in the file.
                return User(**{name: record[name] for name in User.model_fields if name in record})
            except ValidationError as error:
                raise SupportDataError("malformed_record", f"{wanted} is missing required fields") from error

    raise SupportDataError("user_not_found", f"there is no user {wanted}")


def find_user_tickets(user_id: str, data_dir: Path = DATA_DIR) -> UserTickets:
    wanted = _check(USER_ID, user_id, "invalid_user_id")
    find_user(wanted, data_dir)
    summaries: list[TicketSummary] = []

    for record in _records("tickets.json", "tickets", data_dir):
        if not isinstance(record, dict) or str(record.get("user_id", "")).upper() != wanted:
            continue

        try:
            summaries.append(TicketSummary(**{name: record[name] for name in TicketSummary.model_fields if name in record}))
        except ValidationError:
            continue  # a broken record of this user's is skipped, not invented

    return UserTickets(user_id=wanted, tickets=summaries[:MAX_TICKETS_PER_USER])


server = MCPServer(
    SERVER_NAME,
    version=SERVER_VERSION,
    instructions="Read-only access to synthetic support users and tickets, one record at a time.",
)


def _tool_error(error: SupportDataError) -> ToolError:
    return ToolError(str(error))


@server.tool(title="Get a support ticket")
def get_ticket(ticket_id: str) -> Ticket:
    """One support ticket by its id (TICKET-<digits>): status, category, subject, description,
    platform, app version and the troubleshooting steps already tried."""
    try:
        return find_ticket(ticket_id)
    except SupportDataError as error:
        raise _tool_error(error) from error


@server.tool(title="Get a support user")
def get_user(user_id: str) -> User:
    """The support-relevant facts about one user (USER-<digits>): preferred language and plan."""
    try:
        return find_user(user_id)
    except SupportDataError as error:
        raise _tool_error(error) from error


@server.tool(title="List a user's tickets")
def get_user_tickets(user_id: str) -> UserTickets:
    """Summaries of one user's tickets (USER-<digits>) - only that user's."""
    try:
        return find_user_tickets(user_id)
    except SupportDataError as error:
        raise _tool_error(error) from error


if __name__ == "__main__":
    server.run("stdio")
