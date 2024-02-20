PIConGPU Commit Rulez
=====================

We agree on the following simple rules to make our lives easier :)

- Stick to the **style** below for **commit messages**
- **Commit compiling patches** for the `dev` branch,
  you can be less strict for (unshared) *topic branches*
- Follow the :ref:`code style and formatting rules <development-styleguide>` which are democratically evolved in
  [Contributing](https://github.com/ComputationalRadiationPhysics/contributing).

Pre-commit
----------

A good approximation of our guidelines can be easily achieved via [pre-commit](https://pre-commit.com/) hooks (based on
the more general but harder to use [git hooks](https://git-scm.com/docs/githooks)). After the following installation
this little tool will run a number of checks prior to every commit, reject the commit, if they don't succeed, and
potentially apply fixes.

`pre-commit` is a Python tool, so you need a working version of `python3`, e.g. from
[conda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html), your favorite package manager or
[directly from the website](https://www.python.org/downloads/). For installation, we provide
`requirements_pre-commit.txt` and you can use

```bash
# set up a virtual environment if you prefer, then:
$ python3 -m pip install -r requirements_pre-commit.txt

# run this inside of your local clone of the repo:
$ pre-commit install
```

From now on, each commit in this clone of the repo will be checked. See [pre-commit](https://pre-commit.com/) for usage
details. Some hints:

- You can run all hooks on all files via `pre-commit run --all-files`.
- If a check fails, oftentimes it will automatically apply the fixes (e.g. run `clang-format`). If there are no unstaged
  files, these changes will appear as unstaged changes in your working tree. To make the commit pass, you have to `git
  add` all changed files.
- In urgent cases, you can skip the checks via `git commit [...] --no-verify`. Be aware that similar things will be
  checked in CI during your PR and fail then at latest.

Manually Formatting Code
------------------------

For C++ code, we provide `.clang-format` file in the root directory. Python code must adhere to [PEP
8](https://peps.python.org/pep-0008/) guidelines. Following both of these is automated in `pre-commit`. If you are not
able or willing to use `pre-commit`, you can instead do the following manually to get close to the same result:

For Python code, install [`black`](https://pypi.org/project/black/) and run

```bash
black -l 120
```

The following describes formatting of C++ code.

- Install *ClangFormat 12* from LLVM 12.0.1
- To format all files in your working copy, you can run this command in bash from the root folder of PIConGPU:
  ```bash
  find include/ share/picongpu/ share/pmacc -iname "*.def" \
  -o -iname "*.h" -o -iname "*.cpp" -o -iname "*.cu" \
  -o -iname "*.hpp" -o -iname "*.tpp" -o -iname "*.kernel" \
  -o -iname "*.loader" -o -iname "*.param" -o -iname "*.unitless" \
  | xargs clang-format-12 -i
  ```
 
Instead of using the bash command above you can use *Git* together with *ClangFormat* to format your patched code only. 
Before applying this command, you must extend your local git configuration **once** with all file endings used in *PIConGPU*:

```
git config --local clangFormat.extensions def,h,cpp,cu,hpp,tpp,kernel,loader,param,unitless
```

For only formatting lines you added using `git add`, call `git clang-format-12` before you create a commit.
Please be aware that un-staged changes will not be formatted.

Commit Messages
---------------

Let's go for an example:

> Use the 1st line as a topic, stay <= 50 chars
> 
> - the blank line between the "topic" and this "body" is MANDATORY
> - use several key points with - or * for additional information
> - stay <= 72 characters in this "body" section
> - avoid blank lines in the body
> - Why? Pls refer to http://stopwritingramblingcommitmessages.com/


Compile Tests
-------------

We provide an (interactive/automated) **script** that **compiles all examples**
within the `examples/` directory in your branch.

This helps a lot to **maintain various combinations** of options in the code
(like different solvers, boundary conditions, ...).

[![PIConGPU CompileTest](http://img.youtube.com/vi/5b8Xz9nI-hA/0.jpg)](http://www.youtube.com/watch?v=5b8Xz9nI-hA)

Assume
- `repo=<pathToYourPIConGPUgitDirectory>`
- `tmpPath=<tmpFolder>`

Now run the tests with
- `$repo/compile -l $repo/examples/ $tmpPath`

Further options are:
- `-q    : continue on errors`
- `-j <N> : run <N> tests in parallel (note: do NOT omit the number <N>)`

If you ran your test with, let's say `-l -q -j 4`, and you got errors like
>  [compileSuite] [error] In PIC_EXTENSION_PATH:PATH=.../params/KelvinHelmholtz/cmakePreset_0:
>                         CMAKE_INSTALL_PREFIX:PATH=.../params/KelvinHelmholtz/cmakePreset_0
>                         (.../build) make install

check the specific test's output (in this case `examples/KelvinHelmholtz` with
*CMake preset #0*) with:
- `less -R $tmpPath/build/build_ThermalTest_cmakePreset_0/compile.log`


### Compile Tests - Single Example

Compile **all CMake presets** of a *single example* with:
- `$repo/compile $repo/examples/ $tmpPath`


### Compile Tests - Cluster Example:

- Request an interactive job (to release some load from the head node)
  `qsub -I -q laser -lwalltime=03:00:00 -lnodes=1:ppn=64`
- Use a non-home directory, e.g.
  `tmpPath=/net/cns/projects/HPL/<yourTeam>/<yourName>/tmp_tests/`
- Compile like a boss!
  `<pathToYourPIConGPUgitDirectory>/compile -l -q -j 60 <pathToYourPIConGPUgitDirectory>/examples/ $tmpPath`
- Wait for the **thumbs up/down** :)
