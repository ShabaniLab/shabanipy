# Contributing to shabanipy
This document contains the bare minimum requirements for contributing to
shabanipy.
For more information, reach out to one of the [authors](AUTHORS).

## git hooks
Install git hooks with [pre-commit](https://pre-commit.com/):
```shell
conda install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push
```
This will enforce code style conventions on `git commit` and run unit tests on
`git push`.
