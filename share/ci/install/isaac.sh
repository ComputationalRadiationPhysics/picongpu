#!/bin/bash

set -e
set -o pipefail

# merge the PR to the latest version of the destination branch

cd $CI_PROJECT_DIR

export GLM_ROOT=/opt/glm/0.9.9.8
export CMAKE_PREFIX_PATH=$GLM_ROOT/cmake/glm:$CMAKE_PREFIX_PATH
mkdir -p $GLM_ROOT
cd $GLM_ROOT
git clone --branch 0.9.9.8 https://github.com/g-truc/glm.git .

# patch glm to support CUDA compile with clang
sed -i 's/inline/GLM_FUNC_DECL/g' $GLM_ROOT/glm/gtc/type_ptr.inl

cd $CI_PROJECT_DIR
git clone https://github.com/ComputationalRadiationPhysics/isaac.git
cd isaac
git checkout 822f3c240a3bbdf3962984f4fc43f2ea7e75f2eb
mkdir build_isaac
cd build_isaac
cmake ../lib/ -DCMAKE_INSTALL_PREFIX=$ISAAC_ROOT
make install
cd ..
