"""Login gate for the Streamlit app.

Credentials come from the environment (`AUTH_USERNAME`/`AUTH_PASSWORD`, or `AUTH_USERS`
for more than one account) rather than a database, matching how every other secret in
this app is configured. If none are set, the app stays open — this is an opt-in gate,
not a hard requirement, so existing local/demo deployments keep working unchanged.
"""

from __future__ import annotations

import hmac

import streamlit as st

from config import Config


def _credentials(cfg: Config) -> dict[str, str]:
    creds: dict[str, str] = {}
    if cfg.auth_password:
        creds[cfg.auth_username] = cfg.auth_password
    for pair in cfg.auth_users.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        user, _, pw = pair.partition(":")
        user, pw = user.strip(), pw.strip()
        if user and pw:
            creds[user] = pw
    return creds


def enabled(cfg: Config) -> bool:
    return bool(_credentials(cfg))


def _check(username: str, password: str, creds: dict[str, str]) -> bool:
    expected = creds.get(username)
    if expected is None:
        # Compare against something anyway so a nonexistent user takes the same time
        # as a wrong password, rather than leaking valid usernames via timing.
        hmac.compare_digest(password, "")
        return False
    return hmac.compare_digest(password, expected)


def require_login(cfg: Config) -> bool:
    """Render a login form if needed. Returns True once the caller may render the app."""
    if not enabled(cfg):
        return True
    if st.session_state.get("auth_user"):
        return True

    creds = _credentials(cfg)
    st.title("Sign in")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if _check(username, password, creds):
            st.session_state["auth_user"] = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    return False


def render_logout() -> None:
    user = st.session_state.get("auth_user")
    if not user:
        return
    with st.sidebar:
        st.caption(f"Signed in as **{user}**")
        if st.button("Sign out", width="stretch", key="auth_logout"):
            del st.session_state["auth_user"]
            st.rerun()
