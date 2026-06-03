## Summary

What does this PR change, and why?

Fixes #<!-- issue number, if any -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Build / CI / tooling
- [ ] Refactor (no behavior change)

## Checklist

- [ ] Builds and `ctest` passes with **`MATSAR_WITH_BLAS=ON`**
- [ ] Builds and `ctest` passes with **`MATSAR_WITH_BLAS=OFF`**
- [ ] Added or updated tests covering the change (a case that fails before, passes after)
- [ ] The default (experimental-off) build is unaffected
- [ ] Code is formatted (`clang-format`) and follows the project conventions
      ([CONTRIBUTING.md](../CONTRIBUTING.md))
- [ ] Updated docs (`README` / `docs/usage.md` / `CHANGELOG.md`) if behavior or API changed
- [ ] All commits are signed off (DCO: `Signed-off-by`, e.g. `git commit -s`)

## Notes for reviewers

Anything worth calling out — tricky areas, the row-wise / column-major boundary,
platform-specific concerns, follow-ups intentionally left out.
