# shabanipy
Data analysis and plots for Shabani Lab

# Install 
To install the package you can use the python package manager `pip`. First make sure you are at `shabanipy/` directory, then run the following command
```python
pip install . 
```

# For developers

## git hooks
Install git hooks with pre-commit:
```shell
conda install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push
```
This will enforce code style conventions on `git commit` and run unit tests on
`git push`.
