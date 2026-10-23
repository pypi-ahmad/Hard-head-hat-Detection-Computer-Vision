"""Build an installation archive with the selected, hashed SHWD checkpoint."""

import argparse
import hashlib
import json
import tomllib
import zipfile
from pathlib import Path

from safety.inference import selected_checkpoint, selected_resolution
from safety.paths import ROOT


def build(output, caddy_archive):
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    checkpoint, confidence = selected_checkpoint("shwd")
    if checkpoint is None:
        raise ValueError("A selected SHWD checkpoint is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"HelmetCCTV-{version}-windows-x64.zip"
    if archive.exists():
        raise ValueError("Release archive already exists; preserve published builds")
    paths = set(ROOT.glob("*.py"))
    for directory in ("safety", "tests", "docs", "scripts", "licenses"):
        paths.update(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and path.suffix in (".py", ".md", ".ps1", ".json", ".txt"))
    for name in ("README.md", "AGENTS.md", "CLAUDE.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
                 "CHANGELOG.md", "TEST_REPORT.md", "pyproject.toml", "uv.lock", ".python-version",
                 ".streamlit/config.toml", "config/cameras.example.yaml", "Install.cmd", "Start.cmd"):
        paths.add(ROOT / name)
    prefix = f"HelmetCCTV-{version}/"
    hashes = {}
    caddy_archive = Path(caddy_archive)
    with caddy_archive.open("rb") as handle:
        caddy_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if caddy_hash != "0a1edc0b799512051c57071ce0e798d3f2cf67dc98171366f1d2b326072e3b06":
        raise ValueError("Caddy archive checksum mismatch")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        def write(name, content):
            bundle.writestr(prefix + name, content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        for path in sorted(paths):
            if "__pycache__" not in path.parts:
                write(path.relative_to(ROOT).as_posix(), path.read_bytes())
        write("artifacts/shwd/model.pt", checkpoint.read_bytes())
        with zipfile.ZipFile(caddy_archive) as caddy:
            write("vendor/caddy/caddy.exe", caddy.read("caddy.exe"))
            write("vendor/caddy/LICENSE", caddy.read("LICENSE"))
        write("artifacts/shwd/selected_model.json", json.dumps({
            "checkpoint": "artifacts/shwd/model.pt", "confidence": confidence,
            "imgsz": selected_resolution("shwd"), "profile": "shwd",
        }, indent=2).encode())
        manifest = {"version": version, "files": hashes,
            "model_source": checkpoint.relative_to(ROOT).as_posix(),
            "model_limit": "Public SHWD image metrics; deployment-camera/event accuracy is unqualified"}
        bundle.writestr(prefix + "release-manifest.json", json.dumps(manifest, indent=2))
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    checksums = output / "SHA256SUMS.txt"
    checksums.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "sha256": digest, "bytes": archive.stat().st_size}))
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist")
    parser.add_argument("--caddy", default="artifacts/vendor/caddy/caddy_2.11.7_windows_amd64.zip")
    args = parser.parse_args()
    build(args.output, args.caddy)
