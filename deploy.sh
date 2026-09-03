#!/bin/sh

# --- CONFIGURATION ---
PROJECT_DIR="/Users/maxim/nvr"
PLIST_NAME="com.user.ac.plist"
PLIST_SOURCE="$PROJECT_DIR/$PLIST_NAME"
PLIST_DEST="/Library/LaunchDaemons/$PLIST_NAME"
NEWSYSLOG_NAME="com.user.ac.newsyslog.conf"
NEWSYSLOG_SOURCE="$PROJECT_DIR/$NEWSYSLOG_NAME"
NEWSYSLOG_DEST="/etc/newsyslog.d/$NEWSYSLOG_NAME"
# Pinned to an exact interpreter, not the floating `python3` symlink:
# PytorchWildlife's yolov5 dependency doesn't work on Python 3.12 (see
# requirements.txt / README System Requirements), so an unrelated
# `port upgrade python3` on this machine must not silently break the
# daemon by repointing python3 -> 3.12.
PYTHON_BIN="/opt/local/bin/python3.10"
SERVICE_ID="system/com.user.ac"

echo "--- Starting Deployment for Animals Catcher ---"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: $PYTHON_BIN not found. Install it (e.g. 'sudo port install python310') before deploying."
    exit 1
fi

# 1. Pull latest code from GitHub
echo "[1/6] Pulling latest changes from GitHub..."
cd "$PROJECT_DIR" || { echo "Error: Project directory not found"; exit 1; }
git pull origin main

# 2. Sync the .plist to the system folder if changed
if ! diff -q "$PLIST_SOURCE" "$PLIST_DEST" > /dev/null 2>&1; then
    echo "[2/6] Updating system launcher (.plist)..."
    sudo cp "$PLIST_SOURCE" "$PLIST_DEST"
    sudo chown root:wheel "$PLIST_DEST"
    sudo chmod 644 "$PLIST_DEST"
else
    echo "[2/6] Launcher (.plist) is already up to date."
fi

# 3. Sync the newsyslog rotation config if changed. Picked up
# automatically by macOS's own newsyslog service; no reload needed.
if ! diff -q "$NEWSYSLOG_SOURCE" "$NEWSYSLOG_DEST" > /dev/null 2>&1; then
    echo "[3/6] Updating log rotation config..."
    sudo cp "$NEWSYSLOG_SOURCE" "$NEWSYSLOG_DEST"
    sudo chown root:wheel "$NEWSYSLOG_DEST"
    sudo chmod 644 "$NEWSYSLOG_DEST"
else
    echo "[3/6] Log rotation config is already up to date."
fi

# 4. Ensure the interpreter is code-signed. com.user.ac.plist uses
# UserName to drop privileges to an unprivileged account; a completely
# unsigned interpreter running as that dropped-privilege, session-less
# daemon UID gets its outbound network silently blocked by macOS (root
# and real interactive sessions are unaffected regardless of signature,
# which made this very hard to diagnose the first time it happened).
# MacPorts doesn't sign its builds, and a `port upgrade`/reinstall wipes
# any signature applied here, so re-sign on every deploy rather than
# once -- an ad-hoc signature is a no-op to reapply if already present.
echo "[4/6] Ensuring interpreter is code-signed..."
sudo codesign -f -s - "$PYTHON_BIN"

# 5. Ensure Python dependencies are installed (pinned versions -- the
# full pinned set in requirements.txt, including numpy and setuptools,
# is verified to resolve cleanly in one shot; see its comments for why
# each pin is there).
echo "[5/6] Checking Python dependencies..."
$PYTHON_BIN -m pip install -q -r "$PROJECT_DIR/requirements.txt"

# 6. Restart the system service
echo "[6/6] Restarting Animals Catcher Daemon..."
# Attempt to restart the existing service first
if sudo launchctl list | grep -q "com.user.ac"; then
    sudo launchctl kickstart -k system/com.user.ac
else
    # Only bootstrap if it's not loaded at all
    sudo launchctl bootstrap system "$PLIST_DEST"
fi

echo "--- Deployment Successful ---"
