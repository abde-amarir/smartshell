import re
from typing import Optional

# ── Commands where arguments carry meaning and should be KEPT ─────────────────
# e.g. "git checkout main" is different from "git checkout dev"
KEEP_ARGS = {
    "git",
    "docker",
    "systemctl",
    "service",
    "apt",
    "pip",
    "python3",
    "python",
    "sudo",
}

# ── Commands that should be stripped to base only ─────────────────────────────
# e.g. "cd ~/anything" → "cd"
STRIP_TO_BASE = {
    "cd",
    "ls",
    "cat",
    "nano",
    "vim",
    "nvim",
    "code",
    "rm",
    "mv",
    "cp",
    "mkdir",
    "touch",
    "chmod",
    "chown",
    "find",
    "grep",
    "tail",
    "head",
    "less",
    "more",
}

# ── Commands to exclude from training data entirely ───────────────────────────
BLACKLIST = {
    "rm", "dd", "mkfs", "shutdown", "reboot",
    "poweroff", "halt", "mkswap", "fdisk", "parted", 
    "history", "clear", "exit", "logout",
}


def normalize(command: str) -> Optional[str]:
    """
    Normalize a raw command string into a canonical form for model training.

    Returns None if the command should be excluded from training data.

    Examples:
        "cd /var/log"                 → "cd"
        "git commit -m 'fix bug'"     → "git commit"
        "ls -la /home"                → "ls"
        "python3 manage.py runserver" → "python3 manage.py"
        "sudo apt update"             → "sudo apt update"
        "clear"                       → None  (excluded)
    """
    if not command or not command.strip():
        return
    
    command = command.strip()

    # Remove leading shell decorators (time, env vars, etc.)
    # e.g. "FLASK_ENV=dev python3 app.py" → "python3 app.py"
    command = re.sub(r'^[A-Z_]+=\S+\s', '', command)

    tokens = command.split()
    if not tokens:
        return None
    
    base = tokens[0]

    # Handle sudo -- treat "sudo <cmd>" as a unit
    if base == "sudo" and len(tokens) > 1:
        base = tokens[1]
        tokens = tokens[1:]     # shift so logic below applies to real command

    # Blacklist check
    if base in BLACKLIST:
        return None
    
    # Strpi to base command only
    if base in STRIP_TO_BASE:
        return base
    
    # Keep first two tokens for known multi-token commands
    # e.g. "git commit -m ..." → "git commit"
    if base in KEEP_ARGS and len(tokens) > 1:
        return f"{tokens[0]} {tokens[1]}"
    
    # Default — keep just the base command
    return base

def normalize_directory(directory: str) -> str:
    """
    Reduce a full directory path to its last 2 components.
    Prevents overfitting to exact paths.

    Examples:
        "/home/abdessamad/smartshell"  → "abdessamad/smartshell"
        "/var/log"                     → "var/log"
        "/home/abdessamad"             → "home/abdessamad"
    """
    parts = [p for p in directory.strip("/").split("/") if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else directory