#!/bin/bash

set -e
set -o pipefail

github_group_repo="ComputationalRadiationPhysics/picongpu"

pr_id=$(echo "$CI_BUILD_REF_NAME" | cut -d"/" -f1 | cut -d"-" -f2)
# used a token without any rights from psychocoderHPC to avoid API query limitations
curl_data=$(curl -u psychocoderHPC:$GITHUB_TOKEN -X GET https://api.github.com/repos/${github_group_repo}/pulls/${pr_id} 2>/dev/null)
# get the destination branch
all_labels=$(echo "$curl_data" | python3 -c 'import json,sys;obj=json.loads(sys.stdin.read());x = obj["labels"];labels = list(i["name"] for i in x); print(labels)')
echo "search for label: '$1'" >&2
echo "labels: '${all_labels}'" >&2
label_found=$(echo "$all_labels" | grep -q "$1" && echo 0 || echo 1)

exit $label_found
