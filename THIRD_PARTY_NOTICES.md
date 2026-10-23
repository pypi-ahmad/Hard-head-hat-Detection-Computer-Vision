# Third-party components

The repository's existing source license is [MPL 2.0](LICENSE). Dependencies and
model components retain their own terms; that license does not replace theirs.

- **Ultralytics 8.4.184 and YOLO26:** the installed open-source distribution uses
  [AGPL 3.0](licenses/Ultralytics-AGPL-3.0.txt). The release includes a YOLO26 Small
  checkpoint fine-tuned on SHWD. Ultralytics documents its
  [open-source and enterprise options](https://docs.ultralytics.com/help/FAQ/).
  Dependency source is available from the
  [upstream repository](https://github.com/ultralytics/ultralytics/tree/v8.4.184).
- **SHWD:** model training data originated from the
  [Safety Helmet Wearing Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset).
  Its repository's [MIT notice](licenses/SHWD-MIT.txt) is included. Dataset images
  and supplied local videos are excluded from the installation package.
- **Caddy 2.11.7:** the Windows HTTPS gateway uses Apache 2.0. Its license is included
  at `vendor/caddy/LICENSE` in the app ZIP. The
  [upstream source](https://github.com/caddyserver/caddy/tree/v2.11.7) and
  [release checksums](https://github.com/caddyserver/caddy/releases/tag/v2.11.7)
  identify the bundled binary.

`uv.lock` identifies the other Python dependencies and their download hashes.
They are installed from the locked distributions with their own license metadata.
No third-party training dataset is added to the app ZIP.
