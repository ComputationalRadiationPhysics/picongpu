# Contributing

## Formatting

Please format your code before before opening pull requests using clang-format 16 and the .clang-format file placed in the repository root.

### Visual Studio and CLion
Suport for clang-format is built-in since Visual Studio 2017 15.7 and CLion 2019.1.
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
