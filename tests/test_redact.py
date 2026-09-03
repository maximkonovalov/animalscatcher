"""Tests for ac._redact -- added in v0.7 after noticing that requests
exceptions (ConnectionError, Timeout) often stringify the full request
URL, which embeds the Telegram bot token, risking it being written to
ac_log.txt in plaintext."""
import ac


def test_redact_strips_token_from_url_like_exception_text():
    text = (f"HTTPSConnectionPool(host='api.telegram.org'): Max retries "
            f"exceeded with url: /bot{ac.TELEGRAM_TOKEN}/sendMessage")

    redacted = ac._redact(text)

    assert ac.TELEGRAM_TOKEN not in redacted
    assert "<redacted>" in redacted
    # Everything else in the message should survive untouched.
    assert "api.telegram.org" in redacted
    assert "sendMessage" in redacted


def test_redact_leaves_text_without_token_unchanged():
    text = "Connection timed out"

    assert ac._redact(text) == text


def test_redact_handles_empty_token_without_mangling_text(monkeypatch):
    # str.replace(old="", new=...) inserts `new` between every character
    # if not guarded -- this is the regression test for that footgun.
    monkeypatch.setattr(ac, "TELEGRAM_TOKEN", "")

    text = "some ordinary error message"

    assert ac._redact(text) == text


def test_redact_replaces_every_occurrence():
    token = ac.TELEGRAM_TOKEN
    text = f"first {token} and again {token}"

    redacted = ac._redact(text)

    assert token not in redacted
    assert redacted.count("<redacted>") == 2
