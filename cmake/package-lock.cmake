# CPM Package Lock
# This file should be committed to version control

# alpaka
CPMDeclarePackage(alpaka
  NAME alpaka
  # This is a development version slightly after 1.2.0 because we needed a patch
  GIT_TAG 95c0bf2397255a89467bb5c151a96367ad1d1f93
  GITHUB_REPOSITORY alpaka-group/alpaka
  OPTIONS
    "alpaka_CXX_STANDARD 20;alpaka_INSTALL ON"
  # It is recommended to let CPM cache dependencies in order to reduce redundant downloads.
  # However, we might in the foreseeable future turn to unstable references like the `dev` branch here.
  # Setting the following option tells CPM to not use the cache.
  # This is particularly important for CI!
  # NO_CACHE TRUE
)
# cmake-scripts
CPMDeclarePackage(cmake-scripts
  GIT_TAG 24.04
  GITHUB_REPOSITORY StableCoder/cmake-scripts
  SYSTEM YES
  EXCLUDE_FROM_ALL YES
)
# Catch2
CPMDeclarePackage(Catch2
  VERSION 3.7.0
  GITHUB_REPOSITORY catchorg/Catch2
  SYSTEM YES
  EXCLUDE_FROM_ALL YES
)
# Gallatin
CPMDeclarePackage(Gallatin
  # There's no release available yet.
  GIT_TAG ac0cb8e380ffcb74156bafb8805fb60412817c5f
  # Use our own fork for some patches
  GITHUB_REPOSITORY chillenzer/Gallatin
  SYSTEM YES
  EXCLUDE_FROM_ALL YES
)
