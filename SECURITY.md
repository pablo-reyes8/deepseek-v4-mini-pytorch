# Security Policy

## Supported Versions

This project is pre-1.0. Security fixes target the current `main` branch.

## Reporting a Vulnerability

Please open a private security advisory or contact the maintainers directly before publishing details.

Do not include secrets, private datasets, model checkpoints with sensitive data, or credentials in issues, pull requests, configs, notebooks, or logs.

## Dependency Hygiene

- Keep dependencies minimal and pinned through project metadata where possible.
- Review Dependabot updates before merging.
- Treat downloaded datasets and model artifacts as untrusted input.
