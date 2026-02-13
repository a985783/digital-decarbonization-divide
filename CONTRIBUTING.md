# Contributing

Language / 语言: [English](CONTRIBUTING.md) | [中文](CONTRIBUTING.zh-CN.md)

Thanks for helping improve this replication package.

## Development Workflow

1. Create a feature branch.
2. Keep changes focused (one logical change per PR).
3. Run local checks before opening a PR:

```bash
make test
make verify
```

## Coding Guidelines

- Keep analysis changes deterministic and reproducible.
- Do not hardcode local absolute paths.
- Keep generated artifacts out of commits unless they are part of a tagged release payload.

## Pull Request Checklist

- [ ] Repro steps in `REPRODUCIBILITY.md` still work.
- [ ] `scripts/preflight_release_check.py` passes.
- [ ] `scripts/academic_consistency_guard.py` passes.
- [ ] Added/updated tests or documented why not needed.
