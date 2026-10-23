# Security policy

## CCTV installations

The supported LAN setup uses a Caddy HTTPS gateway with hashed-password login;
Streamlit stays on `127.0.0.1:8503`. Camera URLs come from the server environment.
The LAN gateway is intended for the private network described in the installation
guide. Logon startup runs under the configured Windows user account. A locally
trusted certificate does not replace checking that the address belongs to that PC.
Keep `config/*.local*`, `.streamlit/secrets.toml`, recordings, and private PKI keys
outside Git and shared release archives. Trust the local CA only after checking
its fingerprint with the installation administrator. Firewall guidance is in
[CCTV_INSTALL.md](docs/CCTV_INSTALL.md).

## Supported versions

Maintainers apply security fixes to the default branch and recent updates that are
still maintained.

## Reporting a vulnerability

Report potential vulnerabilities through GitHub Issues in this repository.

When reporting, include:

- A clear description of the issue
- Impact assessment
- Reproduction steps or proof of concept
- Suggested remediation, if available

Avoid posting exploit details that could put users at risk. Maintainers may request
additional details privately through GitHub as needed.

## Response process

Maintainers review reports and assess their severity. They share progress through
issue updates and release/change notes when fixes are available.
