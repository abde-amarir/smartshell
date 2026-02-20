import os
import sys
import subprocess
from pathlib import Path
from engine.predictor import get_prediction

# ── Safety Blacklist ───────────────────────────────────────────────────────────
# These commands will NEVER be suggested regardless of model output
BLACKLIST = {
    "rm", "dd", "mkfs", "shutdown", "reboot",
    "poweroff", "halt", "mkswap", "fdisk", "parted",
}


def is_safe(command: str) -> bool:
    """Return False if the command matches a blacklisted base command."""
    base = command.strip().split()[0]
    return base not in BLACKLIST


def display_suggestion(session_id: str = None) -> None:
    """
    Fetch a prediction and display it to the user.
    
    - If the user presses Enter → print the accepted command to stdout
      so the shell can read and execute it.
    - If the user presses ESC or q → silently skip.
    - If no prediction exists → silently exit.
    """
    prediction = get_prediction(session_id=session_id)

    if not prediction:
        sys.exit(0)

    suggested = prediction["suggested"]
    probability = prediction["probability"]
    last_cmd = prediction["last_command"]

    # Safety check — never suggest blacklisted commands
    if not is_safe(suggested):
        sys.exit(0)

    # Only show suggestion if confidence is above threshold
    CONFIDENCE_THRESHOLD = 0.30
    if probability < CONFIDENCE_THRESHOLD:
        sys.exit(0)

    # ── Display ───────────────────────────────────────────────────────────────
    # Write to stderr so it doesn't interfere with stdout capture
    sys.stderr.write(f"\n💡 Suggested: \033[1;36m{suggested}\033[0m")
    sys.stderr.write(f"  \033[2m({probability:.0%} confidence)\033[0m\n")
    sys.stderr.write(  "   \033[2mEnter to accept, any other key to skip\033[0m ")
    sys.stderr.flush()

    # ── Input capture ─────────────────────────────────────────────────────────
    try:
        import tty
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    except Exception:
        sys.stderr.write("\n")
        sys.exit(0)

    sys.stderr.write("\n")

    # Enter key = \r or \n
    if key in ("\r", "\n"):
        sys.stderr.write(f"\033[2m  ✓ Accepted: {suggested}\033[0m\n")
        # Print the command to stdout — the shell will read this
        print(suggested)
    else:
        sys.stderr.write(f"\033[2m  ✗ Skipped\033[0m\n")

    sys.stderr.flush()


if __name__ == "__main__":
    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    display_suggestion(session_id=session_id)