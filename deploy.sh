#!/bin/sh

# --- CONFIGURATION ---
PROJECT_DIR="/Users/maxim/nvr"
PLIST_NAME="com.user.ltsmini.plist"
PLIST_SOURCE="$PROJECT_DIR/$PLIST_NAME"
PLIST_DEST="/Library/LaunchDaemons/$PLIST_NAME"
PYTHON_BIN="/opt/local/bin/python3"
SERVICE_ID="system/com.user.ltsmini"

echo "--- Starting Deployment for LTS-Mini ---"

# 1. Pull latest code from GitHub
echo "[1/4] Pulling latest changes from GitHub..."
cd "$PROJECT_DIR" || { echo "Error: Project directory not found"; exit 1; }
git pull origin main

# 2. Sync the .plist to the system folder if changed
if ! diff -q "$PLIST_SOURCE" "$PLIST_DEST" > /dev/null 2>&1; then
    echo "[2/4] Updating system launcher (.plist)..."
    sudo cp "$PLIST_SOURCE" "$PLIST_DEST"
    sudo chown root:wheel "$PLIST_DEST"
    sudo chmod 644 "$PLIST_DEST"
else
    echo "[2/4] Launcher (.plist) is already up to date."
fi

# 3. Ensure Python dependencies are installed
echo "[3/4] Checking Python dependencies..."
$PYTHON_BIN -m pip install -q opencv-python requests configparser pytorchwildlife

# 4. Restart the system service
echo "[4/4] Restarting LTS-Mini Daemon..."
# Attempt to restart the existing service first
if sudo launchctl list | grep -q "com.user.ltsmini"; then
    sudo launchctl kickstart -k system/com.user.ltsmini
else
    # Only bootstrap if it's not loaded at all
    sudo launchctl bootstrap system "$PLIST_DEST"
fi

echo "--- Deployment Successful ---"
