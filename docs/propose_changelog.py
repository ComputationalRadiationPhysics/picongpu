#!/usr/bin/env python
#
# Copyright 2017-2021 Axel Huebl
#
# License: GPLv3+
#
# requirements:
#   PyGithub
#   curses-menu
#
from github import Github
from cursesmenu import SelectionMenu

# config
#          see: https://github.com/settings/tokens
g = Github("add token with public repo read rights here")
org = g.get_organization("ComputationalRadiationPhysics")
repo = org.get_repo("picongpu")
milestones = repo.get_milestones(sort="asc", state="all")

m_list = list(map(lambda m: m.title, milestones))
menu = SelectionMenu(m_list, "Select a Milestone")
menu.show()
menu.join()
m_sel = menu.selected_option

print(m_list[m_sel], milestones[m_sel].number)

# get pulls (all pulls are also issues but not vice versa)
issues = repo.get_issues(state="closed", sort="updated", direction="asc",
                         milestone=milestones[m_sel])
# Use this in the future, but pagination seems broken in it:
# search_string = 'repo:ComputationalRadiationPhysics/picongpu ' +
#                 'type:pr is:merged milestone:"'+milestones[m_sel].title+'"'
# print(search_string)
# issues = g.search_issues(search_string)


# categories
user = {
    "input": []
}
bugs = {
    "core": [],
    "pmacc": [],
    "plugin": [],
    "tools": [],
    "other": []
}
features = {
    "core": [],
    "pmacc": [],
    "plugin": [],
    "tools": [],
    "other": []
}
refactoring = {
    "core": [],
    "pmacc": [],
    "plugin": [],
    "tools": [],
    "other": []
}
misc = {
    "docs": [],
    "other": []
}

for i in issues:
    # skip issues, only pull requests
    if i.pull_request is None:
        continue

    # filter out bugs that only appeared in development
    pr_nr = i.number
    pr_labels = i.labels
    pr_labels_names = list(map(lambda l: l.name, pr_labels))
    pr_url = "https://github.com/ComputationalRadiationPhysics/picongpu/pull/"
    if "bug" in pr_labels_names:
        if "affects latest release" not in pr_labels_names:
            print("Filtering out development-only bug:")
            print("  #" + str(pr_nr) + " " + i.title)
            print("  " + pr_url + str(pr_nr))
            continue

    # filter out closed (unmerged) PRs
    pr = repo.get_pull(pr_nr)
    if not pr.merged:
        print("Filtering out unmerged PR:")
        print("  #" + str(pr_nr) + " " + i.title)
        print("  " + pr_url + str(pr_nr))
        continue

    # sort by categories
    pr_title = i.title
    if "component: user input" in pr_labels_names:
        user["input"].append(i.title + " #" + str(pr_nr))
    if "affects latest release" in pr_labels_names:
        if "component: core" in pr_labels_names:
            bugs["core"].append(i.title + " #" + str(pr_nr))
        elif "component: PMacc" in pr_labels_names:
            bugs["pmacc"].append(i.title + " #" + str(pr_nr))
        elif "component: plugin" in pr_labels_names:
            bugs["plugin"].append(i.title + " #" + str(pr_nr))
        elif "component: tools" in pr_labels_names:
            bugs["tools"].append(i.title + " #" + str(pr_nr))
        else:
            bugs["other"].append(i.title + " #" + str(pr_nr))
        continue
    if "refactoring" in pr_labels_names:
        if "component: core" in pr_labels_names:
            refactoring["core"].append(i.title + " #" + str(pr_nr))
        elif "component: PMacc" in pr_labels_names:
            refactoring["pmacc"].append(i.title + " #" + str(pr_nr))
        elif "component: plugin" in pr_labels_names:
            refactoring["plugin"].append(i.title + " #" + str(pr_nr))
        elif "component: tools" in pr_labels_names:
            refactoring["tools"].append(i.title + " #" + str(pr_nr))
        else:
            refactoring["other"].append(i.title + " #" + str(pr_nr))
        continue
    # all others are features
    if "component: core" in pr_labels_names:
        features["core"].append(i.title + " #" + str(pr_nr))
        continue
    elif "component: PMacc" in pr_labels_names:
        features["pmacc"].append(i.title + " #" + str(pr_nr))
        continue
    elif "component: plugin" in pr_labels_names:
        features["plugin"].append(i.title + " #" + str(pr_nr))
        continue
    elif "component: tools" in pr_labels_names:
        features["tools"].append(i.title + " #" + str(pr_nr))
        continue
    # all leftovers are miscellaneous changes
    if "documentation" in pr_labels_names:
        misc["docs"].append(i.title + " #" + str(pr_nr))
        continue
    misc["other"].append(i.title + " #" + str(pr_nr))

print("")
print("**User Input Changes:**")
for p in user["input"]:
    print(" - " + p)

print("")
print("**New Features:**")
print(" - PIC:")
for p in features["core"]:
    print("   - " + p)
print(" - PMacc:")
for p in features["pmacc"]:
    print("   - " + p)
print(" - plugins:")
for p in features["plugin"]:
    print("   - " + p)
print(" - tools:")
for p in features["tools"]:
    print("   - " + p)
for p in features["other"]:
    print(" - " + p)

print("")
print("**Bug Fixes:**")
print(" - PIC:")
for p in bugs["core"]:
    print("   - " + p)
print(" - PMacc:")
for p in bugs["pmacc"]:
    print("   - " + p)
print(" - plugins:")
for p in bugs["plugin"]:
    print("   - " + p)
print(" - tools:")
for p in bugs["tools"]:
    print("   - " + p)
for p in bugs["other"]:
    print(" - " + p)

print("")
print("**Misc:**")
print(" - refactoring:")
print("   - PIC:")
for p in refactoring["core"]:
    print("     - " + p)
print("   - PMacc:")
for p in refactoring["pmacc"]:
    print("     - " + p)
print("   - plugins:")
for p in refactoring["plugin"]:
    print("     - " + p)
print("   - tools:")
for p in refactoring["tools"]:
    print("     - " + p)
for p in refactoring["other"]:
    print("   - " + p)
print(" - documentation:")
for p in misc["docs"]:
    print("   - " + p)
for p in misc["other"]:
    print(" - " + p)
