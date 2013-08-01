PIConGPU Commit Rulez
=====================

We agree on the following simple rules to make our lives easier :)

- Stick to the style below for commit messages
- Commit compiling patches for the main branches (trunk/master and dev)


Commit Messages
---------------

Let's go for an example:

Use the 1st line as a topic, stay <= 50 chars

- the blank line between the "topic" and this "body" is MANDATORY
- use several key points with - or * for additional information
- stay <= 72 characters in this "body" section
- avoid blank lines in the body
- Why? Pls refer to http://stopwritingramblingcommitmessages.com/


Compile Tests
-------------

We provide an (interactive/automated) script that compiles all examples
within the examples/ folder in your branch.

This helps a lot to maintain various combinations of options in the code.

Assume
$ repo=<pathToYourPIConGPUsvn>
$ tmpPath=<tmpFolder>

Now run the tests with
$ $repo/compile -l $repo/examples/ $tmpPath

Further options are:
-q     : continue on errors
-j <N> : run <N> tests in parallel (note: do NOT omit the number <N>)

If you ran your test with, let's say "-l -q -j 4", and you got errors like
  [compileSuite] [error] In PIC_EXTENSION_PATH:PATH=.../params/TermalTest/cmakePreset_0:
                         CMAKE_INSTALL_PREFIX:PATH=.../params/TermalTest/cmakePreset_0
                         (.../build) make install
check the specific test's output (in this case examples/TermalTest with CMake Preset #0) with
$ less -R $tmpPath/build/build_TermalTest_cmakePreset_0/compile.log


### Compile Tests - Single Example

Compile all CMake Presets of a single example with:
$ $repo/compile $repo/examples/ $tmpPath


### Compile Tests - Hypnos Example:

- Request an interactive job (to release some load from the head node)
  qsub -I -q laser -lwalltime=03:00:00 -lnodes=1:ppn=64
- Use a non-home directory, e.g.
  tmpPath=/net/cns/projects/HPL/<yourTeam>/<yourName>/tmp_tests/
- Compile like a boss!
  <pathToYourPIConGPUsvn>/compile -l -q -j 60 <pathToYourPIConGPUsvn>/examples/ $tmpPath
- Wait for the thumbs up/down :)
