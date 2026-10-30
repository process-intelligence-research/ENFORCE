## What changes

<!-- What this pull request changes and why. Link the issue it addresses (e.g. "Closes #12"). -->

## Effect on users

<!-- API changes, behaviour changes, or "none". Do results (predictions, residuals, benchmark metrics) change? By how much? -->

## Validation

- [ ] `uv run pytest` passes
- [ ] Tests added or updated for the new behaviour
- [ ] Changed files pass `uv run ruff check`
- [ ] Benchmark reproducibility tests run (`uv run pytest -m slow ...`), if the change can affect training results
