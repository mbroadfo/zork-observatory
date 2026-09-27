"""Where the Claude player's credential comes from.

There is no "sign in with Anthropic" to offer here. The SDK can *consume* an
OAuth credential — a `user_oauth` profile under the config directory, which it
will refresh on its own — but it cannot mint one: no authorize URL, no PKCE, no
device flow. Those profile files are the output of an interactive login
performed by something else, and a browser flow would need an OAuth client
registered to this app, which is not on offer to third parties. Reusing a
client id issued to a different program, or reading the tokens that program
stored, is not a shortcut this project is going to take.

So the two honest paths, in the order they are checked:

  1. The environment — `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`. The
     documented mechanism, and the one that wins if it is set.
  2. A key saved from the browser, in this checkout. This exists because the
     alternative was: stop, edit a file, restart the container, find the page
     again. Pasting it into the setup panel is one action, takes effect on the
     next turn, and needs no restart.

And one that costs nothing to support: if the SDK's own resolution chain finds
a profile on disk, that is reported as configured and used as-is. Nothing here
has to change on the day a login tool exists.

What is saved is a secret in a file, in plain text, in a working directory. On
POSIX it is written 0600; on Windows that is close to meaningless. It is
gitignored, it is never sent anywhere except to the Anthropic API, and it is
never echoed back to the page — a status carries the last four characters so a
reader can tell which key is installed, and nothing more. Anyone who can read
the directory can read the key, and the panel says so rather than implying a
safety it does not have.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from typing import Any

# Bind-mounted in the container so a key saved from the page survives a restart
# and a rebuild; a plain directory when the observatory runs on the host.
SECRET_DIR = ".secrets"
SECRET_FILE = "anthropic.json"

ENV_KEY = "ANTHROPIC_API_KEY"
ENV_TOKEN = "ANTHROPIC_AUTH_TOKEN"
ENV_BASE_URL = "ANTHROPIC_BASE_URL"
# Overrides where the saved key lives, for anyone who keeps secrets elsewhere.
ENV_KEY_FILE = "OBSERVATORY_KEY_FILE"

CONSOLE_URL = "https://console.anthropic.com/settings/keys"

# Long enough that a truncated paste is caught, short enough not to encode an
# assumption about a format Anthropic may change.
MIN_KEY_CHARS = 20


# Variables where "set to the empty string" is worse than "absent". An empty
# ANTHROPIC_BASE_URL is taken by the SDK as the base URL, which then builds a
# request URL with no protocol and reports it as a connection error — sending
# the reader to look at their network for a mistake in their compose file.
EMPTY_IS_WORSE_THAN_ABSENT = (ENV_KEY, ENV_TOKEN, ENV_BASE_URL)


def sanitize_environment() -> list[str]:
    """Drop Anthropic variables that are present but empty. Returns their names.

    `${VAR:-}` in a compose file, an unset secret in a k8s manifest and a blank
    line in a .env all produce this, and none of them mean "use the empty
    string". Called once at startup so the failure cannot reach an agent.
    """
    dropped = []
    for var in EMPTY_IS_WORSE_THAN_ABSENT:
        if var in os.environ and not os.environ[var].strip():
            del os.environ[var]
            dropped.append(var)
    return dropped


def secret_path(root: Path | None = None) -> Path:
    override = os.environ.get(ENV_KEY_FILE)
    if override:
        return Path(override)
    return (root or Path(".")) / SECRET_DIR / SECRET_FILE


def tail(key: str) -> str:
    """The last four characters, for telling two keys apart. Never the key."""
    return f"…{key[-4:]}" if len(key) >= 4 else "…"


def saved(root: Path | None = None) -> str | None:
    """The key written from the page, if there is one."""
    path = secret_path(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    key = data.get("api_key") if isinstance(data, dict) else None
    return key.strip() if isinstance(key, str) and key.strip() else None


def env_credential() -> tuple[str, str] | None:
    """(variable name, value) for whichever environment credential is set."""
    for var in (ENV_KEY, ENV_TOKEN):
        value = os.environ.get(var)
        if value and value.strip():
            return var, value.strip()
    return None


def profile() -> str | None:
    """What the SDK's own resolution chain finds on disk, if anything.

    Asked through the public entry point rather than by guessing at file
    layouts, so an OAuth profile written by some future login tool is picked up
    without a change here. Any failure means "nothing usable", which is the
    only thing this answer is used for.
    """
    try:
        from anthropic import default_credentials

        found = default_credentials()
    except Exception:
        return None
    if found is None:
        return None
    name = type(getattr(found, "provider", found)).__name__
    return name


def resolve(root: Path | None = None) -> tuple[str, str | None]:
    """(source, key). `key` is None when the SDK will find it for itself.

    The environment first, because it is the documented mechanism and the one a
    reader can see. Then a key saved from the page. A profile is left to the
    SDK, which knows how to refresh it and will do so per request.
    """
    found = env_credential()
    if found:
        return found[0], None
    key = saved(root)
    if key:
        return "saved", key
    if profile():
        return "profile", None
    if os.environ.get(ENV_BASE_URL):
        return ENV_BASE_URL, None
    return "none", None


def status(root: Path | None = None) -> dict[str, Any]:
    """What the page needs to decide what to show. Carries no secret."""
    source, key = resolve(root)
    path = secret_path(root)
    out: dict[str, Any] = {
        "present": source != "none",
        "source": source,
        "console_url": CONSOLE_URL,
        "path": str(path),
        "writable": writable(root),
        "detail": "",
    }
    if source == ENV_KEY:
        out["detail"] = f"{ENV_KEY} in the environment"
    elif source == ENV_TOKEN:
        out["detail"] = f"{ENV_TOKEN} in the environment"
    elif source == "saved":
        saved_at = _saved_at(root)
        when = time.strftime("%Y-%m-%d", time.localtime(saved_at)) if saved_at else ""
        out["detail"] = f"a key saved here{f' on {when}' if when else ''}"
        out["tail"] = tail(key or "")
    elif source == "profile":
        out["detail"] = "an Anthropic profile on this machine, which the SDK refreshes itself"
    elif source == ENV_BASE_URL:
        out["detail"] = f"{ENV_BASE_URL} — a gateway is expected to supply the credential"
    else:
        out["detail"] = "nothing configured"
    return out


def _saved_at(root: Path | None = None) -> float | None:
    try:
        data = json.loads(secret_path(root).read_text(encoding="utf-8"))
        value = data.get("saved")
        return float(value) if value else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def writable(root: Path | None = None) -> bool:
    """Whether a key could be saved, asked without creating anything.

    A page that offers to save and then fails on the write is worse than one
    that says up front that this checkout is read-only.
    """
    directory = secret_path(root).parent
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    return os.access(directory, os.W_OK)


def check(key: str) -> str | None:
    """Why this string cannot be a credential, or None. Shape only."""
    if not key or not key.strip():
        return "no key given"
    stripped = key.strip()
    if any(c.isspace() for c in stripped):
        return "that contains whitespace — it looks like more than just the key"
    if len(stripped) < MIN_KEY_CHARS:
        return f"that is only {len(stripped)} characters — a truncated paste?"
    return None


def save(key: str, root: Path | None = None) -> dict[str, Any]:
    """Write the key for this checkout. Refuses anything malformed."""
    problem = check(key)
    if problem:
        return {"ok": False, "detail": problem}
    stripped = key.strip()
    path = secret_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"api_key": stripped, "saved": time.time()}, indent=2) + "\n",
            encoding="utf-8",
        )
        # Best effort, and only that: on Windows this does close to nothing,
        # which is why the panel says plainly where the key is kept.
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:
        return {"ok": False, "detail": f"could not write {path}: {exc}"}
    return {"ok": True, "detail": f"saved to {path}", "tail": tail(stripped)}


def forget(root: Path | None = None) -> dict[str, Any]:
    """Remove the saved key. Anything in the environment is untouched."""
    path = secret_path(root)
    if not path.exists():
        return {"ok": True, "detail": "nothing saved here"}
    try:
        path.unlink()
    except OSError as exc:
        return {"ok": False, "detail": f"could not remove {path}: {exc}"}
    return {"ok": True, "detail": f"removed {path}"}
