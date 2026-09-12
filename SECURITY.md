# Security

MiniTorch is a **local educational library**. It does not talk to the network,
does not start services, and does not load cloud credentials.

## What this project is

- NumPy-backed autodiff, training, compression, and experiment tooling
- Artifact I/O for versioned `.mtc` files and JSON experiment results
- Checksums (CRC32) for artifact integrity, not for authentication

## What this project is not

- A production inference server
- A secret store
- A remote experiment tracker

`.env` is unused at runtime. [`.env.example`](.env.example) exists only so
contributors do not invent a secrets layout.

## Reporting a vulnerability

If you find a security issue (unsafe deserialization, path traversal in
artifact loaders, or anything that could execute untrusted data):

1. Prefer [GitHub Security Advisories](https://docs.github.com/en/code-security/security-advisories) on the repository.
2. If advisories are unavailable, open a private report with the maintainers
   or an issue **without** a working exploit.

Please include the affected file, MiniTorch version (`0.1.0`), and a
minimal reproduction.

## Safe handling of artifacts

- Treat downloaded `.mtc` / experiment JSON as untrusted input.
- Loaders must reject malformed schema, magic, or checksums. They must not
  `exec` or `pickle` user files.
- Do not store API keys or passwords inside model artifacts.

## Supported versions

| Version | Support |
| --- | --- |
| 0.1.x | Current educational release |
