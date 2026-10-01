#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

# ============================================================
# APPY Common Launcher - macOS / Linux
# App: PhaseEQ
# ============================================================
APP_NAME="PhaseEQ"
MAIN_FILE="phaseeq.py"
COMPOSITE_FILE="composite_streamlit_app.py"
BASE_PORT=8501
MAX_PORT=8520
STARTUP_RETRY_COUNT=60
STARTUP_RETRY_DELAY_SECONDS=0.3
STARTUP_REQUEST_TIMEOUT_SECONDS=1.0
VENV_DIR=".venv"
LOG_DIR="logs"
ENABLE_LOG=0
FORCE_UPDATE=0
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=12
MAX_PYTHON_MAJOR=3
MAX_PYTHON_MINOR=14
PYTHON_VERSION_FILE="$VENV_DIR/.launcher-python-version"

for argument in "$@"; do
    case "$argument" in
        --log)
            ENABLE_LOG=1
            ;;
        --update)
            FORCE_UPDATE=1
            ;;
        --help|-h)
            echo "Usage: $0 [--log] [--update]"
            echo "  default  Start offline when the existing runtime is complete"
            echo "  --update Update pip and dependencies before starting (network required)"
            echo "  --log    Write launcher output to logs/${APP_NAME}_launcher.log"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $argument"
            echo "Usage: $0 [--log] [--update]"
            exit 2
            ;;
    esac
done

main() {
    echo "== $APP_NAME Launcher =="

    if [ ! -f "$MAIN_FILE" ]; then
        echo "ERROR: $MAIN_FILE was not found."
        exit 1
    fi
    if [ ! -f "$COMPOSITE_FILE" ]; then
        echo "ERROR: $COMPOSITE_FILE was not found."
        exit 1
    fi

    find_python
    check_python_version
    if [ -f ".phaseeq-updates/pending.json" ] || [ -f ".phaseeq-updates/journal.json" ]; then
        if "$PYTHON" runtime/app_update.py --apply; then
            :
        else
            update_status=$?
            if [ "$update_status" -eq 10 ]; then
                exec sh "$0" "$@"
            fi
            return "$update_status"
        fi
    fi
    check_python_venv
    ensure_venv

    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate"
    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

    python -c "import sys; print('Python:', sys.executable)"
    python --version

    if [ ! -f "requirements.txt" ]; then
        echo "ERROR: requirements.txt was not found."
        exit 1
    fi

    RUNTIME_ACTION="$(python runtime/check_runtime_environment.py --action)"
    if [ "$FORCE_UPDATE" -eq 1 ]; then
        echo "Online update requested."
        prepare_runtime_repair "$RUNTIME_ACTION"
        install_dependencies
        check_japanese_fonts 1
    elif [ "$RUNTIME_ACTION" = "ready" ] && runtime_dependencies_ready; then
        echo "Runtime dependencies are ready. Starting without network access."
        check_japanese_fonts 0
    else
        echo "Runtime dependencies require action: $RUNTIME_ACTION"
        echo "Online installation is required for this setup."
        prepare_runtime_repair "$RUNTIME_ACTION"
        install_dependencies
        check_japanese_fonts 1
    fi

    if ! runtime_dependencies_ready; then
        echo "ERROR: Runtime dependency verification failed."
        exit 1
    fi

    if [ "${PHASEEQ_LAUNCHER_CHECK_ONLY:-0}" = "1" ]; then
        echo "Launcher environment check completed."
        return
    fi

    PORT="$(find_free_port)"
    ORIGINAL_BASE_PORT="$BASE_PORT"
    BASE_PORT=$((PORT + 1))
    COMPOSITE_PORT="$(find_free_port)"
    BASE_PORT="$ORIGINAL_BASE_PORT"
    export PORT
    export PHASEEQ_SERVER_PORT="$PORT"
    export COMPOSITE_ENGINE_PORT="$COMPOSITE_PORT"
    export PHASEEQ_URL="http://localhost:$PORT"
    export COMPOSITE_ENGINE_URL="http://localhost:$COMPOSITE_PORT"
    export PHASEEQ_COMPOSITE_EXCHANGE_DIR="${PHASEEQ_DATA_DIR:-$PWD/data}/tmp/composite_exchange"

    echo
    echo "Starting $APP_NAME..."
    echo "URL: http://localhost:$PORT"
    echo "Composite Engine: http://localhost:$COMPOSITE_PORT"
    echo "Network: local-only (external web features remain optional)"
    echo

    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    python -m streamlit run "$COMPOSITE_FILE" \
        --server.port "$COMPOSITE_PORT" \
        --server.address 127.0.0.1 \
        --server.headless true &
    COMPOSITE_PID=$!
    python -m streamlit run "$MAIN_FILE" \
        --server.port "$PORT" \
        --server.address 127.0.0.1 \
        --server.headless true &
    PHASEEQ_PID=$!
    trap cleanup_servers EXIT INT TERM

    if ! wait_for_servers; then
        return 1
    fi

    open_browser "http://localhost:$PORT"
    open_browser "http://localhost:$COMPOSITE_PORT"

    wait "$PHASEEQ_PID"
}

cleanup_servers() {
    if [ -n "${PHASEEQ_PID:-}" ]; then
        kill "$PHASEEQ_PID" >/dev/null 2>&1 || true
    fi
    if [ -n "${COMPOSITE_PID:-}" ]; then
        kill "$COMPOSITE_PID" >/dev/null 2>&1 || true
    fi
}

http_ready() {
    health_url="$1"
    python - "$health_url" "$STARTUP_REQUEST_TIMEOUT_SECONDS" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
timeout = float(sys.argv[2])
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        raise SystemExit(0 if 200 <= response.status < 400 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_servers() {
    phaseeq_health_url="http://127.0.0.1:$PORT/_stcore/health"
    composite_health_url="http://127.0.0.1:$COMPOSITE_PORT/_stcore/health"
    attempt=1

    echo "Waiting for PhaseEQ and Composite Engine to become ready..."
    while [ "$attempt" -le "$STARTUP_RETRY_COUNT" ]; do
        if ! kill -0 "$PHASEEQ_PID" >/dev/null 2>&1; then
            echo "ERROR: PhaseEQ stopped before becoming ready."
            return 1
        fi
        if ! kill -0 "$COMPOSITE_PID" >/dev/null 2>&1; then
            echo "ERROR: Composite Engine stopped before becoming ready."
            return 1
        fi

        phaseeq_ready=0
        composite_ready=0
        if http_ready "$phaseeq_health_url"; then
            phaseeq_ready=1
        fi
        if http_ready "$composite_health_url"; then
            composite_ready=1
        fi
        if [ "$phaseeq_ready" -eq 1 ] && [ "$composite_ready" -eq 1 ]; then
            echo "PhaseEQ and Composite Engine are ready."
            return 0
        fi

        if [ "$attempt" -lt "$STARTUP_RETRY_COUNT" ]; then
            sleep "$STARTUP_RETRY_DELAY_SECONDS"
        fi
        attempt=$((attempt + 1))
    done

    echo "ERROR: Startup timed out after $STARTUP_RETRY_COUNT attempts."
    echo "PhaseEQ health: $phaseeq_health_url"
    echo "Composite Engine health: $composite_health_url"
    return 1
}

runtime_dependencies_ready() {
    python runtime/check_runtime_environment.py >/dev/null 2>&1 || return 1
    python runtime/check_runtime_environment.py --optional-warnings
    python -c "import altair, numpy, pandas, scipy, soundfile, streamlit; import composite_engine" >/dev/null 2>&1
}

prepare_runtime_repair() {
    action="$1"
    if [ "$action" = "recreate" ]; then
        echo "Multiple incompatible or broken packages were found."
        command -v deactivate >/dev/null 2>&1 && deactivate || true
        recreate_venv
        # shellcheck disable=SC1091
        . "$VENV_DIR/bin/activate"
    fi
}

install_dependencies() {
    TOO_NEW_PACKAGES="$(python runtime/check_runtime_environment.py --too-new)"
    if [ -n "$TOO_NEW_PACKAGES" ]; then
        echo "Removing packages newer than the supported environment: $TOO_NEW_PACKAGES"
        # Package names are emitted from the repository's validated constraints.
        # shellcheck disable=SC2086
        python -m pip uninstall -y $TOO_NEW_PACKAGES
    fi
    echo "Updating pip..."
    python -m pip install --constraint constraints/test-environment.txt pip setuptools wheel
    echo "Installing the latest validated runtime dependencies..."
    python -m pip install --upgrade --constraint constraints/test-environment.txt -r requirements.txt
}

find_python() {
    PYTHON=""

    for candidate in \
        python3.13 \
        python3.12 \
        "$HOME/.pyenv/versions/3.12.8/bin/python" \
        "$HOME/.pyenv/versions/3.12.7/bin/python" \
        python3 \
        python
    do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c "import sys; raise SystemExit(0 if (${MIN_PYTHON_MAJOR}, ${MIN_PYTHON_MINOR}) <= sys.version_info[:2] < (${MAX_PYTHON_MAJOR}, ${MAX_PYTHON_MINOR}) else 1)" >/dev/null 2>&1; then
                PYTHON="$candidate"
                break
            fi
        fi
    done

    if [ -z "$PYTHON" ]; then
        echo "Python 3.12 or 3.13 was not found."
        install_python

        for candidate in \
            python3.13 \
            python3.12 \
            "$HOME/.pyenv/versions/3.12.8/bin/python" \
            "$HOME/.pyenv/versions/3.12.7/bin/python" \
            python3 \
            python
        do
            if command -v "$candidate" >/dev/null 2>&1; then
                if "$candidate" -c "import sys; raise SystemExit(0 if (${MIN_PYTHON_MAJOR}, ${MIN_PYTHON_MINOR}) <= sys.version_info[:2] < (${MAX_PYTHON_MAJOR}, ${MAX_PYTHON_MINOR}) else 1)" >/dev/null 2>&1; then
                    PYTHON="$candidate"
                    break
                fi
            fi
        done
    fi

    if [ -z "$PYTHON" ]; then
        echo "ERROR: Python 3.12 or 3.13 could not be found."
        exit 1
    fi

    export PYTHON
}

install_python() {
    if command -v brew >/dev/null 2>&1; then
        echo "Installing Python by Homebrew..."
        brew update
        brew install python

    elif command -v apt >/dev/null 2>&1; then
        echo "Installing Python by apt / pyenv..."
        install_python_apt_packages

    elif command -v dnf >/dev/null 2>&1; then
        echo "Installing Python by dnf..."
        sudo dnf install -y \
            python3 \
            python3-pip \
            python3-devel \
            python3-virtualenv \
            gcc \
            gcc-c++ \
            make

    elif command -v pacman >/dev/null 2>&1; then
        echo "Installing Python by pacman..."
        sudo pacman -Sy --needed \
            python \
            python-pip \
            base-devel

    else
        echo "ERROR: Supported package manager was not found."
        echo "Please install Python 3.12 or 3.13 manually."
        exit 1
    fi
}

install_python_apt_packages() {
    sudo apt update

    if apt-cache show python3.12 >/dev/null 2>&1; then
        sudo apt install -y \
            python3.12 \
            python3.12-venv \
            python3.12-dev \
            python3-pip \
            build-essential
    else
        echo "python3.12 package was not found in apt."
        echo "Installing Python 3.12.8 by pyenv..."

        sudo apt install -y \
            build-essential \
            curl \
            git \
            libssl-dev \
            zlib1g-dev \
            libbz2-dev \
            libreadline-dev \
            libsqlite3-dev \
            libncursesw5-dev \
            xz-utils \
            tk-dev \
            libxml2-dev \
            libxmlsec1-dev \
            libffi-dev \
            liblzma-dev

        if [ ! -d "$HOME/.pyenv" ]; then
            git clone https://github.com/pyenv/pyenv.git "$HOME/.pyenv"
        fi

        PYENV_ROOT="$HOME/.pyenv"
        export PYENV_ROOT
        export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

        "$PYENV_ROOT/bin/pyenv" install -s 3.12.8
        "$PYENV_ROOT/bin/pyenv" local 3.12.8 || true
    fi
}

check_python_version() {
    if ! "$PYTHON" -c "import sys; raise SystemExit(0 if (${MIN_PYTHON_MAJOR}, ${MIN_PYTHON_MINOR}) <= sys.version_info[:2] < (${MAX_PYTHON_MAJOR}, ${MAX_PYTHON_MINOR}) else 1)"; then
        echo "ERROR: Python 3.12 or 3.13 is required. Python 3.14 is not supported by the validated runtime."
        "$PYTHON" --version
        exit 1
    fi

    "$PYTHON" --version
}

check_python_venv() {
    if "$PYTHON" -m venv --help >/dev/null 2>&1; then
        return
    fi

    echo "Python venv support is missing."
    echo "Trying to install missing venv support..."

    install_python_venv_packages

    if ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
        echo "ERROR: Python venv support is still unavailable."
        exit 1
    fi
}

install_python_venv_packages() {
    if command -v apt >/dev/null 2>&1; then
        install_python_apt_packages

    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y \
            python3-virtualenv \
            python3-devel \
            gcc \
            gcc-c++ \
            make

    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --needed \
            python \
            python-pip \
            base-devel

    elif command -v brew >/dev/null 2>&1; then
        brew update
        brew install python

    else
        echo "ERROR: Could not install Python venv support automatically."
        exit 1
    fi
}

ensure_venv() {
    NEED_RECREATE=0

    if [ ! -d "$VENV_DIR" ]; then
        NEED_RECREATE=1
    elif [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "activate script is missing."
        NEED_RECREATE=1
    elif [ ! -x "$VENV_DIR/bin/python" ]; then
        echo "python executable is missing."
        NEED_RECREATE=1
    elif [ ! -x "$VENV_DIR/bin/pip" ]; then
        echo "pip executable is missing."
        NEED_RECREATE=1
    elif ! "$VENV_DIR/bin/python" -c "import sys" >/dev/null 2>&1; then
        echo "python executable is broken."
        NEED_RECREATE=1
    elif ! "$VENV_DIR/bin/python" -c "import sys; raise SystemExit(0 if (${MIN_PYTHON_MAJOR}, ${MIN_PYTHON_MINOR}) <= sys.version_info[:2] < (${MAX_PYTHON_MAJOR}, ${MAX_PYTHON_MINOR}) else 1)" >/dev/null 2>&1; then
        echo "Virtual environment uses an unsupported Python version."
        "$VENV_DIR/bin/python" --version || true
        NEED_RECREATE=1
    fi

    if [ "$NEED_RECREATE" -eq 1 ]; then
        echo "Virtual environment is missing, broken, or outdated."
        recreate_venv
    else
        # A healthy venv remains usable after the system Python is updated.
        # Record the interpreter actually used by the venv without recreating
        # it or requiring dependency downloads.
        "$VENV_DIR/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' > "$PYTHON_VERSION_FILE"
    fi
}

recreate_venv() {
    echo "Recreating virtual environment..."
    rm -rf "$VENV_DIR"

    if ! "$PYTHON" -m venv "$VENV_DIR"; then
        echo
        echo "Virtual environment creation failed."
        echo "Trying to install missing packages..."
        echo
        install_python_venv_packages
        echo
        echo "Retrying virtual environment creation..."
        echo
        "$PYTHON" -m venv "$VENV_DIR"
    fi

    if [ ! -f "$VENV_DIR/bin/activate" ] ||
        [ ! -x "$VENV_DIR/bin/python" ] ||
        [ ! -x "$VENV_DIR/bin/pip" ]; then
        echo
        echo "ERROR: Virtual environment is incomplete."
        exit 1
    fi

    "$VENV_DIR/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' > "$PYTHON_VERSION_FILE"
}

check_japanese_fonts() {
    allow_install="${1:-0}"
    if [ "$(uname -s 2>/dev/null || echo unknown)" = "Darwin" ]; then
        return
    fi

    if ! command -v fc-list >/dev/null 2>&1; then
        echo
        echo "fontconfig was not found."
        if [ "$allow_install" -eq 1 ]; then
            echo "Installing fontconfig and Japanese fonts..."
            echo
            install_japanese_font_packages
            fc-cache -fv >/dev/null 2>&1 || true
        else
            echo "WARNING: Offline start continues without automatic font installation."
            echo "Run $0 --update while online to install optional Japanese fonts."
        fi
        return
    fi

    if fc-list | grep -qiE "Noto Sans CJK|Noto Serif CJK|IPAex|IPAGothic|Yu Gothic|Meiryo"; then
        return
    fi

    echo
    echo "Japanese font was not found."
    if [ "$allow_install" -eq 1 ]; then
        echo "Installing Japanese fonts..."
        echo
        install_japanese_font_packages
        fc-cache -fv >/dev/null 2>&1 || true
    else
        echo "WARNING: Offline start continues without automatic font installation."
        echo "Run $0 --update while online to install optional Japanese fonts."
    fi
}

install_japanese_font_packages() {
    if command -v apt >/dev/null 2>&1; then
        sudo apt update
        sudo apt install -y \
            fontconfig \
            fonts-noto-cjk \
            fonts-ipafont \
            fonts-ipaexfont

    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y \
            fontconfig \
            google-noto-sans-cjk-fonts \
            google-noto-serif-cjk-fonts

    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --needed \
            fontconfig \
            noto-fonts-cjk

    else
        echo "WARNING: Supported package manager was not found."
        echo "Please install a Japanese CJK font such as Noto Sans CJK manually."
    fi
}

find_free_port() {
    p="$BASE_PORT"

    while [ "$p" -le "$MAX_PORT" ]; do
        if command -v lsof >/dev/null 2>&1; then
            if ! lsof -i :"$p" >/dev/null 2>&1; then
                echo "$p"
                return 0
            fi

        elif command -v ss >/dev/null 2>&1; then
            if ! ss -ltn | grep -q ":$p "; then
                echo "$p"
                return 0
            fi

        else
            echo "$p"
            return 0
        fi

        p=$((p + 1))
    done

    echo "ERROR: No free port found from $BASE_PORT to $MAX_PORT." >&2
    exit 1
}

open_browser() {
    url="$1"

    if command -v open >/dev/null 2>&1; then
        open "$url" >/dev/null 2>&1 || true

    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 || true
    fi
}

if [ "$ENABLE_LOG" -eq 1 ]; then
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/${APP_NAME}_launcher.log"

    {
        echo "==== $(date) ===="
        main "$@"
    } >> "$LOG_FILE" 2>&1
else
    main "$@"
fi
