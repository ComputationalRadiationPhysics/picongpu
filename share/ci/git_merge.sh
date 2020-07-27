#!/bin/bash

set -e
set -o pipefail

# merge the PR to the latest version of the destination branch

cd $CI_PROJECT_DIR

github_group_repo="ComputationalRadiationPhysics/picongpu"

pr_id=$(echo "$CI_BUILD_REF_NAME" | cut -d"/" -f1 | cut -d"-" -f2)
# used a token without any rights from psychocoderHPC to avoid API query limitations
curl_data=$(curl -u psychocoderHPC:$GITHUB_TOKEN -X GET https://api.github.com/repos/${github_group_repo}/pulls/${pr_id} 2>/dev/null)
# get the destination branch
destination_branch=$(echo "$curl_data" | python3 -c 'import json,sys;obj=json.loads(sys.stdin.read());print(obj["base"]["ref"])')
destination_sha=$(echo "$curl_data" | python3 -c 'import json,sys;obj=json.loads(sys.stdin.read());print(obj["base"]["sha"])')
echo "destination_branch=${destination_branch}"
echo "destination_sha=${destination_sha}"

mainline_exists=$(git remote -v | cut -f1 | grep mainline -q && echo 0 || echo 1)
# avoid adding the remote repository twice if gitlab already cached this operation
if [ $mainline_exists -ne 0 ] ; then
  git remote add mainline https://github.com/${github_group_repo}.git
else
  # if the PR was set to a different branch before
  git remote set-url mainline https://github.com/${github_group_repo}.git
fi
git fetch mainline

# required by git to be able to use `git rebase`
git config --global user.email "CI-BOT"
git config --global user.name "CI-BOT@hzdr.d"

# make a copy of the pull request branch
git checkout -b pr_to_merge
# switch to the destination hash
git checkout -b destination_branch ${destination_sha}
# merge pull request to the destination
git merge --no-edit pr_to_merge
