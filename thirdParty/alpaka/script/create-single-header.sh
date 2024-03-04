#!/bin/bash

workingcopy_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )/.."
dst=${1:-$workingcopy_dir/single-header/include/alpaka/alpaka.hpp}
tmp_dir=$(mktemp -d)
git clone https://github.com/shrpnsld/amalgamate.git --depth 1 $tmp_dir/clone
$tmp_dir/clone/amalgamate -o $tmp_dir -v -a -n 'alpaka' -I $workingcopy_dir/include -- $workingcopy_dir/include/alpaka/alpaka.hpp
mv $tmp_dir/alpaka-amalgamated/alpaka.hpp $dst
rm -rf $tmp_dir
