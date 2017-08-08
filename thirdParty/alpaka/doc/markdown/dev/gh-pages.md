[:arrow_up: Up](../Index.md)

Publishing doxygen documentation on gh-pages
============================================

To deploy the doxygen documentation a copy of the repository is created inside the deployed folder.
This copy is always in the gh-pages branch consisting only of the containing files.
This folder is ignored in all other branches.

Creation of gh-pages
--------------------

*NOTE:* This has already been done once and does not have to be repeated!

On working branch:
- Add deploy directory to `.gitignore` (if not already done)
- Create the `gh-pages` branch: `git checkout --orphan gh-pages`
- Clean the branch: `git rm -rf .`
- Commit and push the branch: `git add --all`, `git commit -m"add gh-pages branch"`, `git push`

Setup
-----

*NOTE:* This has to be done once per cloned alpaka repository that is used to deploy the doxygen documentation!

On working branch:
- Clone the repo on the gh-pages branch inside the deploy folder: `git clone -b gh-pages git@github.com:ComputationalRadiationPhysics/alpaka.git doc/doxygen/html`

Update
------

From within `develop`/`master`: 
- Execute doxygen
- `cd doc/doxygen/html`
- `git add .`
- `git commit -m "updated doxygen documentation"`
- `git push`
- `cd ../../../`
