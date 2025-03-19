# SETUP GUIDE

## Project Setup with `uv`

We use `uv` so you can get set up with one command in <10 seconds. You don't need conda, pip, or even python installed. `uv` handles everything for you, installing the right python version and packages. If you're used to using `pip`, just run `uv pip ...` to do pip things and to run python files you can do `uv run python ...`.

1. Install `uv` from https://docs.astral.sh/uv/getting-started/installation/
2. Run `uv sync` (from the project directory)

## Alternative Setup

If you don't want to use `uv`, and instead want to manage your virtual envs yourself, we also have a `requirements.txt` file. 
This method is easy enough as well, but remember that our graders are running `uv` for quick spin-up and faster grading. 
Please let us know if you run into trouble.

1. `pip install .`
2. `pip install -e .` (don't forget the dot at the end!)

## Running Files and Tests
1. Run python code with `uv run file.py`
2. Run tests with `uv run tests.py`
3. To run pytests directly use `uv run pytest tests/*_test.py`. While this is an option, we recommend you just run `tests.py` since this formats and colors the output nicely for you.
   1. To see verbose outputs add the `-v` flag
   2. To fail on the first failed test add the `-x` flag. This is very useful since there are a lot of tests.
   3. To show less verbose output with a short summary at the end add the `-q` flag
4. `tests.py` will also generate a `results.json`, showing you how many points you earned
