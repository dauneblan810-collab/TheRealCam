#!/usr/bin/env bash
# Installs everything needed to build the Android app on Linux:
#   JDK 17 (via mise, falls back to apt), Android cmdline-tools, platform 35, build-tools.
# Idempotent. Writes env exports to $HOME/.therealcam-env and android/local.properties.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_DIR="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/android-sdk}}"
JAVA_SPEC="${JAVA_SPEC:-temurin-17}"
ENV_FILE="$HOME/.therealcam-env"

log() { printf '\n[setup] %s\n' "$*"; }

# ---------- JDK 17 ----------
JAVA_HOME_DETECTED=""
if command -v mise >/dev/null 2>&1; then
  log "Installing JDK ($JAVA_SPEC) via mise"
  mise install "java@$JAVA_SPEC" >/dev/null
  JAVA_HOME_DETECTED="$(mise where "java@$JAVA_SPEC")"
elif command -v apt-get >/dev/null 2>&1; then
  log "Installing JDK 17 via apt"
  apt-get update -y >/dev/null && apt-get install -y openjdk-17-jdk-headless >/dev/null
  JAVA_HOME_DETECTED="/usr/lib/jvm/java-17-openjdk-amd64"
fi
[ -x "$JAVA_HOME_DETECTED/bin/java" ] || { echo "JDK 17 not found" >&2; exit 1; }
export JAVA_HOME="$JAVA_HOME_DETECTED"
export PATH="$JAVA_HOME/bin:$PATH"
java -version 2>&1 | head -1

# ---------- Android cmdline-tools ----------
SDKMANAGER="$SDK_DIR/cmdline-tools/latest/bin/sdkmanager"
if [ ! -x "$SDKMANAGER" ]; then
  log "Downloading Android cmdline-tools"
  zip_name="$(curl -fsSL https://dl.google.com/android/repository/repository2-3.xml \
    | grep -o 'commandlinetools-linux-[0-9]*_latest.zip' | sort -u -t- -k3 -n | tail -1)"
  zip_name="${zip_name:-commandlinetools-linux-11076708_latest.zip}"
  tmp_zip="$(mktemp --suffix=.zip)"
  curl -fsSL -o "$tmp_zip" "https://dl.google.com/android/repository/$zip_name"
  mkdir -p "$SDK_DIR/cmdline-tools"
  rm -rf "$SDK_DIR/cmdline-tools/latest" "$SDK_DIR/cmdline-tools/cmdline-tools"
  python3 -m zipfile -e "$tmp_zip" "$SDK_DIR/cmdline-tools"
  mv "$SDK_DIR/cmdline-tools/cmdline-tools" "$SDK_DIR/cmdline-tools/latest"
  chmod +x "$SDK_DIR/cmdline-tools/latest/bin/"*
  rm -f "$tmp_zip"
fi

log "Installing SDK packages"
yes | "$SDKMANAGER" --sdk_root="$SDK_DIR" --licenses >/dev/null 2>&1 || true
"$SDKMANAGER" --sdk_root="$SDK_DIR" "platform-tools" "platforms;android-35" "build-tools;35.0.0" "build-tools;34.0.0" >/dev/null

# ---------- env + local.properties ----------
cat > "$ENV_FILE" <<EOF
export JAVA_HOME="$JAVA_HOME"
export ANDROID_HOME="$SDK_DIR"
export ANDROID_SDK_ROOT="$SDK_DIR"
export PATH="\$JAVA_HOME/bin:\$ANDROID_HOME/platform-tools:\$ANDROID_HOME/cmdline-tools/latest/bin:\$PATH"
EOF
if [ -d "$REPO_ROOT/android" ]; then
  echo "sdk.dir=$SDK_DIR" > "$REPO_ROOT/android/local.properties"
fi
log "Done. Run: source $ENV_FILE"
