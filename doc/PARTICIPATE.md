PIConGPU - How to participate
===============================================================================

to do list:
- [ ] coding guide lines, styles/fileHeaders/...
- [ ] commit rules
- [ ] compile suite movie
- [x] git for svn users
- [ ] explain pull requests

*******************************************************************************

Code - Version Control
----------------------

If you are familiar with git, feel free to jump to our [github workflow](#github-workflow) section.

### git

Git is a *distributed* **version control system**. It helps you to keep your software
development work organized, because it keeps track of *changes* in your project.
It also helps to come along in **teams**, crunching on
the *same project*. Examples:
- Arrr, dare you other guys! Why did you change my precious *main.cpp*, too!?
- Who introduced that awesome block of code? I would like to pay for a beer as a reward.
- Everything is wrong now, why did this happen and when?
- What parts of the code changed since I went on vacation
  (to a conference, phd seminar,
   [mate](http://en.wikipedia.org/wiki/Club-Mate) fridge, ...)?

If *version control* is totally **new** to you (that's good, because you are not
[spoiled](http://www.youtube.com/watch?v=4XpnKHJAok8)) - please refer to a
beginners guide first.
- [git - the simple guide](http://rogerdudler.github.io/git-guide/)
- 15 minutes guide at [try.github.io](http://try.github.io)

Since git is *distributed*, no one really needs a server or services like
github.com to *use git*. Actually, there are even very good reasons why one should
use git even for **local** data, e.g. a master thesis (or your collection of ascii art
dwarf hamster pictures).

Btw, **fun fact warning**: [Linus Torvalds](http://en.wikipedia.org/wiki/Linus_Torvalds),
yes the nice guy with the pinguin stuff and all that, developed git to maintain
the **Linux kernel**. So that's cool, by definition.

A nice overview about the *humongous* number of tutorials can be found at
[stackoverflow.com](http://stackoverflow.com/questions/315911/git-for-beginners-the-definitive-practical-guide)
... but we may like to start with a git **cheat sheet** (is there anyone out there who knows more
than 1% of all git commands available?)
- [git-tower.com](http://www.git-tower.com/files/cheatsheet/Git_Cheat_Sheet_grey.pdf) (print the 1st page)
- [github.com - "cheat git" gem](https://help.github.com/articles/git-cheatsheet) (a cheat sheet for the console)
- [kernel.org](https://www.kernel.org/pub/software/scm/git/docs/everyday.html) *Everyday GIT with 20 commands or so*
- [an other interactive, huge cheat sheet](http://ndpsoftware.com/git-cheatsheet.html)
  (nice overview about stash - workspace - index - local/remote repositories)

Please spend a minute to learn how to write **useful**
[git commit](https://github.com/blog/926-shiny-new-commit-styles)
**messages** (caption-style, maximum characters per line, use blank lines, present tense)

### git for svn users

If you already used version control systems before, you may enjoy the
[git for svn users crash course](http://git.or.cz/course/svn.html).

Anyway, please keep in mind to use git *not* like a centralized version control
system (e.g. *not* like svn). Imagine git as your *own private* svn server
waiting for your commits. For example *Github.com* is only **one out of many** *sources for updates*.
(But of course, we agree to share our *finished*, new features there.)

GitHub Workflow
---------------

Welcome to github! We will try to explain our coordination strategy (I am out
of here!) and our development workflow in this section.

### How to fork from us

To keep our development fast and conflict free, we recomment you to
[fork](https://help.github.com/articles/fork-a-repo) our **dev** (development)
branch into your private repository. Simply click the *Fork* button above to do so.

Afterwards, `git clone` **your** repository to your
[local machine](https://help.github.com/articles/fork-a-repo#step-2-clone-your-fork).
But that is not it! To keep track of the original **dev** repository, add
it as another [upstream](https://help.github.com/articles/fork-a-repo#step-3-configure-remotes).
- `git remote add upstream https://github.com/ComputationalRadiationPhysics/picongpu.git`

Well done so far! Just start developing. Just like this? No! As always in git,
start a fresh branch with `git branch <yourFeatureName>` and apply your changes there.

### Keep track of updates

We consider it a **best practice** *not to modify your master* branch (== our *dev*)
branch at all. Instead you can use it to `pull` new updates from the original
repository and to start **new feature branches** from.

So, if you are really clever, you can even
[keep track](http://de.gitready.com/beginner/2009/03/09/remote-tracking-branches.html)
of the *original dev* branch that way. Just start your new branch with
`git branch --track <yourFeatureName> upstream/dev`
instead. This allows you to immediatly pull or fetch from **our dev** and avoids typing.

You should **add updates** from the original repository on a **regular basis**
or *at least* when you *finished your feature*.
- commit your local changes in your *feature branch*: `git commit`

Now you *could* do a normal *merge* of the latest `upstream/dev` changes into
your feature branch. That is indeed possible, but will create an ugly
[merge commit](http://kernowsoul.com/blog/2012/06/20/4-ways-to-avoid-merge-commits-in-git/).
Instead try to first update *the point where you branched from* and apply
your changes again. That is called a **rebase** and is indeed less harmful as
reading the sentence before:
- `git checkout <yourFeatureName>`
- `git pull --rebase`

Now solve your conflicts, if there are any, and you got it! Well done!

### Pull requests or *being social*

How to propose that **your awesome feature** (we know it will be awesome!) should be
included in the **mainline PIConGPU** version?

...

pull requests & policies

### maintainer notes

- do not *push* to the main repository on a regular basis, use **pull request**
  for your features like everyone else
- **never** do a *rebase* on the upstream repositories
  (this causes heavy problems for everyone who pulls them)
- on the other hand try to use
  [pull --rebase](http://kernowsoul.com/blog/2012/06/20/4-ways-to-avoid-merge-commits-in-git/)
  to **avoid merge commits** (in your local/topic branches)
- do not vote on your *own pull requests*, wait for the other maintainers
- we try to follow the strategy of [a-successful-git-branching-model](http://nvie.com/posts/a-successful-git-branching-model/)

### Last but not least

[help.github.com](https://help.github.com/) has a very
nice FAQ section.

Developers with a Mac may like to visit http://mac.github.com/

*******************************************************************************

coding guide lines
------------------

Well - there are some! ;)
- *coming soon* (in a separate file)

Please **add the according license header** snippet to your *new files*:
- for PIConGPU (GPLv3+): `src/tools/bin/addLicense <FileName>`
- for libraries (LGPLv3+ & GPLv3+):
  `export PROJECT_NAME=libgpugrid && src/tools/bin/addLicense <FileName>`
- delete other headers: `src/tools/bin/deleteHeadComment <FileName>`
- add license to all .hpp files within a directory (recursive):
  `export PROJECT_NAME=PIConGPU && src/tools/bin/findAndDo <PATH> "*.hpp" src/tools/bin/addLicense`
- the default project name ist `PIConGPU` (case sensitive!) and adds the GPLv3+
  only.

*******************************************************************************

Test Suite Examples
-------------------

You know a useful setting to validate our provided methods?
Tell us about it or add it to our test sets in the examples/ folder!

