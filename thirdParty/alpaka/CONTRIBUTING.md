# Contributing

Please review our more detailed [Coding Guidelines](https://alpaka.readthedocs.io/en/latest/dev/style.html) as well.

## Pre-commit

This project is set up for use with [pre-commit](https://pre-commit.com). Using it will make your code conform with most
of our (easily automatable) code style guidelines automatically.
In very short (for anything further see [pre-commit](https://pre-commit.com)), after running the following in your
working clone of alpaka
```bash
# if not yet done, install the pre-commit executable following https://pre-commit.com
cd /path/to/alpaka-working-clone
pre-commit install
```
`git` will run a number of checks prior to every commit and push and will refuse to perform the
pertinent action if they fail. Most of them (like e.g. the formatter) will have automatically altered your working tree
with the necessary changes such that
```bash
git add -u
```
will make the next commit pass.

## Formatting

Please format your code before before opening pull requests using clang-format 16 and the .clang-format file placed in 
the repository root. If you were using `pre-commit` during your changes, this has happened automatically already. If
not, find further instructions below.

### Visual Studio and CLion
Support for clang-format is built-in since Visual Studio 2017 15.7 and CLion 2019.1.
The .clang-format file in the repository will be automatically detected and formatting is done as you type, or triggered when pressing the format hotkey.

### Bash
First install clang-format-16. Instructions therefore can be found on the web.
To format your changes since branching off develop, you can run this command in bash:
```
git clang-format-16 develop
```
To format all code in your working copy, you can run this command in bash:
```
find -iname '*.cpp' -o -iname '*.hpp' | xargs clang-format-16 -i
```
