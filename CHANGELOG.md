# Changelog

All notable changes to this project are documented here.

## [0.1.0] - 2026-09-23

### Added

- Formal Apache-2.0 licensing, attribution, citation, and publication status.
- Hash-based train/validation/test leakage checker with unit tests.
- Reproducible synthetic inference smoke-test artifact.
- Security guidance for credential handling and incident response.

### Changed

- Clarified the research prototype's scope, authorship, contact route, and
  limitations.
- Preserved nested input paths in outputs to prevent filename collisions.
- Hardened inference parameter validation, SAM device handling, and LLM API
  endpoint construction.

### Removed

- Duplicated training/validation image and generated dataset caches.
- Placeholder result figures that could be mistaken for measured results.
- The leaked credential from all reachable Git history. The credential must
  still be revoked at its issuing provider.
