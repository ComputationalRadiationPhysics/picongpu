How to Participate as a Developer
===============================================================================

Contents
--------

1. [Code - Version Control](#code---version-control)
 - [Install git](#install-git)
 - [git](#git)
 - [git for svn users](#git-for-svn-users)
2. [GitHub Workflow](#github-workflow)
 - [In a Nutshell](#in-a-nutshell)
 - [How to Fork From Us](#how-to-fork-from-us)
 - [Keep Track of Updates](#keep-track-of-updates)
 - [Pull Requests or *Being Social*](#pull-requests-or-being-social)
 - [Maintainer Notes](#maintainer-notes)
3. [Commit Rules](#commit-rules)
4. [Test Suite Examples](#test-suite-examples)

*******************************************************************************

Code - Version Control
----------------------

If you are familiar with git, feel free to jump to our [github workflow](#github-workflow) section.

### install git

**Debian/Ubuntu**:
- `sudo apt-get install git`
- make sure `git --version` is at least at version
  [1.7.10](https://help.github.com/articles/https-cloning-errors)

Optional *one* of these. There are nice GUI tools available to get an overview
on your repository.
- `gitk git-gui qgit gitg`

**Mac**:
- see [here](https://help.github.com/articles/set-up-git)

**Windows**:
- see [here](http://lmgtfy.com/?q=debian+-+or+how+to+download+a+real+operating+system)
- just kidding, it's [this link](https://help.github.com/articles/set-up-git)
- please use ASCII for your files and take care of
  [line endings](https://help.github.com/articles/dealing-with-line-endings)

**Configure** your global git settings:
- `git config --global user.name NAME`
- `git config --global user.email EMAIL@EXAMPLE.com`
- `git config --global color.ui "auto"` (if you like colors)
- `git config --global pack.threads "0"` (improved performance for multi cores)

You may even improve your level of awesomeness by:
- `git config --global alias.pr "pull --rebase"`
  (see how to [avoide merge commits](#keep-track-of-updates))
- `git config --global alias.pm "pull --rebase mainline"` (to sync with the mainline by `git pm dev`)
- `git config --global alias.st "status -sb"` (short status version)
- `git config --global alias.l "log --oneline --graph --decorate --first-parent"` (single branch history)
- `git config --global alias.la "log --oneline --graph --decorate --all"` (full branch history)
- `git config --global rerere.enable 1`
  (see [git rerere](https://git-scm.com/blog/2010/03/08/rerere.html))
- More `alias` tricks:
  - `git config --get-regexp alias` (show all aliases)
  - `git config --global --unset alias.<Name>` (unset alias `<Name>`)


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
   [mate](https://en.wikipedia.org/wiki/Club-Mate) fridge, ...)?

If *version control* is totally **new** to you (that's good, because you are not
[spoiled](https://www.youtube.com/watch?v=4XpnKHJAok8)) - please refer to a
beginners guide first.
- [git - the simple guide](http://rogerdudler.github.io/git-guide/)
- 15 minutes guide at [try.github.io](https://try.github.io)

Since git is *distributed*, no one really needs a server or services like
github.com to *use git*. Actually, there are even very good reasons why one should
use git even for **local** data, e.g. a master thesis (or your collection of ascii art
dwarf hamster pictures).

Btw, **fun fact warning**: [Linus Torvalds](https://en.wikipedia.org/wiki/Linus_Torvalds),
yes the nice guy with the pinguin stuff and all that, developed git to maintain
the **Linux kernel**. So that's cool, by definition.

A nice overview about the *humongous* number of tutorials can be found at
[stackoverflow.com](https://stackoverflow.com/questions/315911/git-for-beginners-the-definitive-practical-guide)
... but we may like to start with a git **cheat sheet** (is there anyone out there who knows more
than 1% of all git commands available?)
- [git-tower.com](https://www.git-tower.com/blog/git-cheat-sheet/) (print the 1st page)
- [github.com - "cheat git" gem](https://help.github.com/articles/git-cheatsheet) (a cheat sheet for the console)
- [kernel.org](https://www.kernel.org/pub/software/scm/git/docs/everyday.html) *Everyday GIT with 20 commands or so*
- [an other interactive, huge cheat sheet](http://ndpsoftware.com/git-cheatsheet.html)
  (nice overview about stash - workspace - index - local/remote repositories)

Please spend a minute to learn how to write **useful**
[git commit](https://github.com/blog/926-shiny-new-commit-styles)
**messages** (caption-style, maximum characters per line, use blank lines,
present tense). Read our [commit rules](#commit-rules) and use
[keywords](https://help.github.com/articles/closing-issues-via-commit-messages).

If you like, you can **credit** someone else for your **next commit** with:
- `git commit --author "John Doe <johns-github-mail@example.com>"`

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

### In a Nutshell

Create a *GitHub* account and prepare your [basic git config](#install-git).

Prepare your *forked* copy of our repository:
- fork [picongpu](https://github.com/ComputationalRadiationPhysics/picongpu)
  on *GitHub*
- `git clone git@github.com:<YourUserName>/picongpu.git` (create local copy)
- `git remote add mainline git@github.com:ComputationalRadiationPhysics/picongpu.git`
  (add our main repository for updates)
- `git checkout dev` (switch to our, its now *your*, dev branch to start from)

Start a *topic/feature branch*:
- `git checkout -b <newFeatureName>` (start a new branch from dev and check it out)
- *hack hack*
- `git add <yourChangedFiles>` (add changed and new files to index)
- `git clang-format` (format all changed files with the clang-format utility, needs to be loaded or installed separately)
- `git add <yourChangedFiles>` (add the formating changes) 
- `git commit` (commit your changes to your *local* repository)
- `git pull --rebase mainline dev` (update with our *remote dev* updates and
  avoid a [merge commit](http://kernowsoul.com/blog/2012/06/20/4-ways-to-avoid-merge-commits-in-git/))

Optional, *clean up* your feature branch. That can be *dangerous*:
- `git pull` (if you pushed your branch already to your public repository)
- `git pull --rebase mainline dev` (apply the mainline updates to your feature branch)
- `git log ..mainline/dev`, `git log --oneline --graph --decorate --all` (check for related commits and ugly merge commits)
- `git rebase mainline/dev` (re-apply your changes after a fresh update to the
  `mainline/dev`, see [here](http://git-scm.com/book/en/ch3-6.html))
- `git rebase -i mainline/dev`
  ([squash](http://blog.steveklabnik.com/posts/2012-11-08-how-to-squash-commits-in-a-github-pull-request)
  related commits to reduce the complexity of the features history during a
  [pull request](https://help.github.com/articles/using-pull-requests))

*Publish* your feature and start a *pull request*:
- `git push -u origin <newFeatureName>` (push your local branch to your github
  profile)
- Go to your *GitHub* page and open a *pull request*, e.g. by clicking on
  *compare & review*
- Add additional updates (if requested to do so) by `push`-ing to your branch
  again. This will update the *pull request*.

### How to fork from us

To keep our development fast and conflict free, we recomment you to
[fork](https://help.github.com/articles/fork-a-repo) our repository and start
your work from our **dev** (development)
branch in your private repository. Simply click the *Fork* button above to do so.

Afterwards, `git clone` **your** repository to your
[local machine](https://help.github.com/articles/fork-a-repo/#step-2-create-a-local-clone-of-your-fork).
But that is not it! To keep track of the original **dev** repository, add
it as another [remote](https://help.github.com/articles/fork-a-repo/#step-3-configure-git-to-sync-your-fork-with-the-original-spoon-knife-repository).
- `git remote add mainline https://github.com/ComputationalRadiationPhysics/picongpu.git`
- `git checkout dev` (go to branch **dev**)

Well done so far! Just start developing. Just like this? No! As always in git,
start a *new branch* with `git checkout -b topic-<yourFeatureName>` and apply your
changes there.

### Keep track of updates

We consider it a **best practice** *not to modify* your **dev** branch at all.  
Instead you can use it to `pull --ff-only` new updates from
the original repository. Take care to **switch to dev** by `git checkout dev` to start
**new feature branches** from **dev**.

So, if you like to do so, you can even
[keep track](http://de.gitready.com/beginner/2009/03/09/remote-tracking-branches.html)
of the *original dev* branch that way. Just start your new branch with
`git branch --track <yourFeatureName> mainline/dev`
instead. This allows you to immediatly pull or fetch from **our dev** and
avoids typing (during `git pull --rebase`). Nevertheless, if you like to `push` to
*your* forked (== `origin`) repository, you have to say e.g.
`git push origin <branchName>` explicitly.

You should **add updates** from the original repository on a **regular basis**
or *at least* when you *finished your feature*.
- commit your local changes in your *feature branch*: `git commit`

Now you *could* do a normal *merge* of the latest `mainline/dev` changes into
your feature branch. That is indeed possible, but will create an ugly
[merge commit](http://kernowsoul.com/blog/2012/06/20/4-ways-to-avoid-merge-commits-in-git/).
Instead try to first update *the point where you branched from* and apply
your changes *again*. That is called a **rebase** and is indeed less harmful as
reading the sentence before:
- `git checkout <yourFeatureName>`
- `git pull --rebase mainline dev` (in case of an emergency, hit `git rebase --abort`)

Now solve your conflicts, if there are any, and you got it! Well done!

### Pull requests or *being social*

How to propose that **your awesome feature** (we know it will be awesome!) should be
included in the **mainline PIConGPU** version?

Due to the so called [pull requests](https://help.github.com/articles/using-pull-requests) in *GitHub*, this quite easy (yeah, sure).
We start again with a *forked repository* of our own.
You already created a **new feature branch** starting from our **dev** branch and commited your changes.
Finally, you publish you local branch via a *push* to *your* GitHub repository: `git push -u origin <yourLocalBranchName>`

Now let's start a *review*. Open the *GitHub* homepage, go to your repository and
switch to your *pushed feature branch*. Select the green **compare & review**
button. Now compare the changes between **your feature branch** and **our dev**.

Everything looks good? Submit it as a **pull request** (link in the header).
Please take the time to write an **extensive description**.

- What did you implement and why?
- Is there an open issue that you try to address (please link it)?
- Do not be afraid to add images!

The description of the pull request is essential and will be referred to in the
change log of the next release.

Please consider to change only **one aspect per pull request** (do not be afraid
of follow-up pull requests!). For example, submit a pull request with a bug fix,
another one with new math implementations and the last one with a new awesome
implementation that needs both of them. You will see, that speeds up *review time*
a lot!

Speaking of those, a fruitful ( *wuhu, we love you - don't be scared* ) *discussion*
about your **submitted change set** will start at this point. If we find some things you
could *improve* ( *That looks awesome, all right!* ), simply change your
*local feature branch* and *push the changes back* to your GitHub repository,
to **update the pull request**. (You can now rebase follow-up branches, too.)

One of our [maintainers](README.md#maintainers-and-core-developers)
will pick up the pull request to coordinate the review.
Other regular developers that are competent in the topic might assist.

Sharing is caring! Thank you for participating, **you are great**!

### maintainer notes

- do not *push* to the main repository on a regular basis, use **pull request**
  for your features like everyone else
- **never** do a *rebase* on the mainline repositories
  (this causes heavy problems for everyone who pulls them)
- on the other hand try to use
  [pull --rebase](http://kernowsoul.com/blog/2012/06/20/4-ways-to-avoid-merge-commits-in-git/)
  to **avoid merge commits** (in your *local/topic branches* **only**)
- do not vote on your *own pull requests*, wait for the other maintainers
- we try to follow the strategy of [a-successful-git-branching-model](http://nvie.com/posts/a-successful-git-branching-model/)

Last but not least, [help.github.com](https://help.github.com/) has a very nice
FAQ section.

More [best practices](http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/).

*******************************************************************************

Commit Rules
------------

See our [commit rules page](docs/COMMIT.md)

*******************************************************************************

Test Suite Examples
-------------------

You know a useful setting to validate our provided methods?
Tell us about it or add it to our test sets in the examples/ folder!

