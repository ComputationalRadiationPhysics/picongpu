PIConGPU Commit Rulez
=====================

We agree on the following simple rules to make our lives easier :)

- Stick to the **style** below for **commit messages**
- **Commit compiling patches** for the `dev` branch,
  you can be less strict for (unshared) *topic branches*
- Follow the code style and formatting which is democratically evolved in
  [Contributing](https://github.com/ComputationalRadiationPhysics/contributing).

Pre-commit
----------

Compliance with the coding style can be ensured (as much as automatically possible) by using
[pre-commit](https://pre-commit.com/) hooks (based on the more general but harder to use [git
hooks](https://git-scm.com/docs/githooks)). After the following installation this little tool will run a number of
checks prior to every commit, reject the commit, if they don't succeed, and potentially apply fixes.

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

- You can run all hooks on all files via `pre-commit run --all-files [--hook-stage manual]`. The last argument
  `--hook-stage manual` includes rather slow additional tests that are run by the CI but are considered too heavy-weight
  to run before each commit.
- If a check fails, the fixes are often applied automatically (e.g., run `clang-format`). If there are no unstaged
  files, these changes will appear as unstaged changes in your working tree. To make the commit pass, you have to `git
  add` all changed files.
- In urgent cases, you can skip the checks via `git commit [...] --no-verify`. Be aware that similar things will be
  checked in CI during your PR and fail then at latest.

### Rebasing onto a change of convention/formatting

Updates in conventions and formatting occasionally happen on purpose or due to version updates in the tooling.
If you've started some branch before such changes took place,
they can lead to basically all your changes being marked as conflicts once you try to merge/rebase.
With `pre-commit` being the ground truth for following the conventions,
there's a relatively straightforward way to rectify this.

Let's assume we have a situation where you have a branch `conflicting-changes`
and a `mainline/dev` has a commit `change-formatting` that introduced some formatting changes
by altering `.pre-commit-config.yaml` in the root of the repository.
For simplicity, you can find out the hash of that commit and use

```bash
git branch change-formatting <hash-of-change-formatting>
```

to create a label for reference.

Applying a change in convention is conveniently bundled in the following three commands
that I'd suggest to store in a file `apply-change-of-convention.sh`

```bash
git restore --source change-formatting .pre-commit-config.yaml
pre-commit run --all-files
git restore .pre-commit-config.yaml
```

0. Rebase `conflicting-changes` to `change-formatting~1` (right before `change-formatting`).
You might see real conflicts that you really have to resolve yourself.

```bash
git checkout conflicting-changes
git rebase change-formatting~1
```

1. Now, we start lifting you over the formatting change. Start an interactive rebase

```bash
git rebase -i change-formatting
```

2. Inside of the editor that opened up, mark all commits as `edit`

```
edit 43cebfdceb Conflicting change
edit 75d366bca3 Conflicting change 2
# and so on...
```

3. No patch will apply cleanly because the formatting gets in the way.
But instead of resolving all the conflicts manually,
we know precisely what change to apply:

```bash
git restore --source REBASE_HEAD .
# If you decided against creating that file, paste the lines from above:
bash apply-change-of-convention.sh
git add -u
# make sure you're doing the right thing:
# git diff --staged
git rebase --continue
```

Unfortunately, I haven't found a convenient way to automate this within an (interactive) rebase.
It would surely be doable within a script but having a little more control is probably also nice.

Code Formatting
---------------

C++ code is formatted with clang-format 20.1.8 (LLVM 20.1) according to the
`.clang-format` file in the repository root; Python code follows [PEP
8](https://peps.python.org/pep-0008/) and is formatted with [`black`](https://pypi.org/project/black/)
(line length 120). Both are automated by the pre-commit hooks above.

For manual formatting without pre-commit and for the `git clang-format`
workflow, see {ref}`Coding Guide Lines <development-styleguide>`.

Commit Messages
---------------

Let's go for an example:

> Use the 1st line as a topic, stay <= 50 chars
>
> - the blank line between the "topic" and this "body" is MANDATORY
> - use several key points with - or * for additional information
> - stay <= 72 characters in this "body" section
> - avoid blank lines in the body
> - Why? Pls refer to <http://stopwritingramblingcommitmessages.com/>

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
