# Contributing to ENFORCE

Thanks for your interest in ENFORCE! Bug reports, questions, ideas and pull requests are welcome. Everyone taking
part in the project is expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting help and reporting problems

Use [GitHub issues](https://github.com/process-intelligence-research/ENFORCE/issues) for questions, bug reports and
feature requests. The issue templates ask for what we need to reproduce a problem; for numerical problems
(constraints not satisfied, divergence, NaNs) please include the constraint definition, the tolerance you expected,
the residual you observed and a minimal script that reproduces it.

The project is maintained by [Giacomo Lastrucci](https://github.com/giacomolastrucci) at the
[Process Intelligence Research](https://www.pi-research.org/) group of Artur M. Schweidtmann, TU Delft.

## Development setup

Fork the repository on GitHub ("Fork" button), then clone your fork and add this repository as `upstream`:

```
git clone https://github.com/<your-username>/ENFORCE
cd ENFORCE
git remote add upstream https://github.com/process-intelligence-research/ENFORCE
uv sync
```

Maintainers with write access can clone this repository directly and skip the fork.

`uv sync` installs the package in editable mode with the development tools (pytest, ruff, pyright, coverage) and
the dependencies of the notebooks and benchmarks. PyTorch comes from the CPU index set in `pyproject.toml`; see the
README for CUDA.

## Checks

Fast tests (run in CI on every pull request, a few minutes):

```
uv run pytest
```

Lint and type checks. The existing code does not pass them everywhere yet, so check the files you change:

```
uv run ruff check <changed files>
uv run pyright <changed files>
```

Benchmark reproducibility (slow, from minutes to hours per problem; not run in CI). They need the benchmark datasets
from the link in the README, placed under `data/raw/`:

```
uv run pytest -m slow tests/test_benchmark_reproducibility.py
```

Run them when a change can affect training results (projection, losses, training loop, constraints). They compare
against the saved baselines in `tests/fixtures/benchmark_baselines/`.

## Pull requests

1. Open an issue first for larger changes, so we can agree on the approach.
2. Create a branch from the latest `main` of this repository:
   ```
   git fetch upstream
   git switch -c my-change upstream/main
   ```
3. Keep the change focused, add or update tests for the new behaviour, keep the fast tests passing and the changed
   files clean under ruff.
4. Push the branch to your fork (`git push -u origin my-change`) and open a pull request into `main` of this
   repository. Describe what changes for users and how you validated it (the pull request template asks for this).
5. Address review comments by pushing to the same branch; the pull request updates automatically. If `main` has moved
   on, update your branch with `git fetch upstream` and `git rebase upstream/main`.

CI runs the fast tests on every pull request. For first-time contributors, a maintainer approves the first run.

Numerical changes should not silently change results: if the outputs of the benchmarks or the tests change, say by
how much and why.

By contributing, you agree that your contributions are licensed under the MIT license of the project.
