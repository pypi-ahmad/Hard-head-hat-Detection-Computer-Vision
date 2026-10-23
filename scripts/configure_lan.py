"""Create the local HTTPS gateway config without storing plaintext passwords."""

import getpass
import json
import re
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gateway_config(host, username, password_hash, port=8443, upstream=8503):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]{0,252}", host):
        raise ValueError("Use a PC name, DNS name, or IPv4 address without a port")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", username):
        raise ValueError("Use letters, numbers, hyphens or underscores for the username")
    if not re.fullmatch(r"\$2[aby]\$\d\d\$[./A-Za-z0-9]{53}", password_hash):
        raise ValueError("Invalid bcrypt hash")
    if not all(isinstance(v, int) and 1024 <= v <= 65535 for v in (port, upstream)):
        raise ValueError("Invalid port")
    names = list(dict.fromkeys([host, "localhost", "127.0.0.1"]))
    addresses = ", ".join(f"https://{name}:{port}" for name in names)
    return f'''{{
    admin off
    persist_config off
    auto_https disable_redirects
    skip_install_trust
    storage file_system {{
        root artifacts/cctv/pki
    }}
}}
{addresses} {{
    tls internal
    basic_auth {{
        {username} {password_hash}
    }}
    reverse_proxy 127.0.0.1:{upstream}
}}
'''


def main():
    host = input(f"LAN PC name or fixed IPv4 address [{socket.gethostname()}]: ").strip() or socket.gethostname()
    username = input("Dashboard username [operator]: ").strip() or "operator"
    password = getpass.getpass("Dashboard password (at least 12 characters): ")
    if len(password) < 12 or len(password.encode()) > 72:
        raise ValueError("Use at least 12 characters and at most 72 UTF-8 bytes")
    if getpass.getpass("Confirm password: ") != password:
        raise ValueError("Passwords do not match")
    caddy = ROOT / "vendor/caddy/caddy.exe"
    # Binary input avoids Windows translating LF into CRLF inside the password.
    result = subprocess.run([str(caddy), "hash-password"], input=password.encode() + b"\n",
                            capture_output=True, check=True)
    del password
    config = gateway_config(host, username, result.stdout.decode().strip())
    destination = ROOT / "config/Caddyfile.local"
    destination.write_text(config, encoding="utf-8")
    (ROOT / "config/lan.local.json").write_text(json.dumps({"url": f"https://{host}:8443"}),
                                               encoding="utf-8")
    print(f"LAN dashboard configured: https://{host}:8443")
    print("Trust the installation's root certificate on each client. See docs/CCTV_INSTALL.md.")


if __name__ == "__main__":
    main()
