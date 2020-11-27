PIConGPU Commit Rulez
=====================

We agree on the following simple rules to make our lives easier :)

- Stick to the **style** below for **commit messages**
- **Commit compiling patches** for the *main* branches (`master` and `dev`),
  you can be less strict for (unshared) *topic branches*
- Commits should be formated with clang-format-11

Format Code
-----------

- Install *ClangFormat 11*
- To format all files in your working copy, you can run this command in bash from the root folder of PIConGPU:
  ```bash
  find include/ share/picongpu/ share/pmacc -iname "*.def" \
  -o -iname "*.h" -o -iname "*.cpp" -o -iname "*.cu" \
  -o -iname "*.hpp" -o -iname "*.tpp" -o -iname "*.kernel" \
  -o -iname "*.loader" -o -iname "*.param" -o -iname "*.unitless" \
  | xargs clang-format-11 -i
  ```
 
Instead of using the bash command above you can use *Git* together with *ClangFormat* to format your patched code only. 
Before applying this command, you must extend your local git configuration **once** with all file endings used in *PIConGPU*:

```
git config --local clangFormat.extensions def,h,cpp,cu,hpp,tpp,kernel,loader,param,unitless
```

For only formatting lines you added using `git add`, call `git clang-format-11` before you create a commit.
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
>  [compileSuite] [error] In PIC_EXTENSION_PATH:PATH=.../params/ThermalTest/cmakePreset_0:
>                         CMAKE_INSTALL_PREFIX:PATH=.../params/ThermalTest/cmakePreset_0
>                         (.../build) make install

check the specific test's output (in this case `examples/ThermalTest` with
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

