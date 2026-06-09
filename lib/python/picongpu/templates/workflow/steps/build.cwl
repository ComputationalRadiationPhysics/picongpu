# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

cwlVersion: v1.2
class: CommandLineTool
label: "Build PIConGPU"
doc: "Compile PIConGPU using pic-build with the provided build script"

requirements:
  InitialWorkDirRequirement:
    listing:
      - entryname: include
        entry: $(inputs.include_directory)
      - entryname: build.sh
        entry: $(inputs.script)

baseCommand: ./build.sh

inputs:
  include_directory:
    type: Directory
    label: "Compile-time parameter header directory"
    doc: "Directory containing compile-time parameter headers for compilation of PIConGPU"
  script:
    type: File
    label: "Build script"
    doc: "Shell script setting up the environment and running pic-build"
  jobs:
    type: int?
    inputBinding:
      position: 2
      prefix: "-j"
    label: "Number of parallel jobs"
    doc: "Allow N jobs at once"
    default: 4
  cmake:
    type: string?
    inputBinding:
      position: 3
      prefix: "-c"
    label: "Extra CMake arguments"
    doc: "Extra arguments passed straight to CMake"
    default: null
  preset:
    type: int?
    inputBinding:
      position: 4
      prefix: "-t"
    label: "CMake preset number"
    doc: "Configure this preset number from CMake flags"
    default: null
  force:
    type: boolean
    inputBinding:
      position: 5
      prefix: "-f"
    label: "Force rebuild"
    doc: "Clear CMake cache and force scan for new .param files"
    default: false
  cmake_build_system:
    type: string?
    inputBinding:
      position: 6
      prefix: "-G"
    label: "CMake build system"
    doc: "Select the build system used by CMake (e.g. Ninja)"
    default: null
outputs:
  bin_directory:
    type: Directory
    outputBinding:
      glob: "bin"
    label: "Compiled executables"
    doc: "Compiled PIConGPU binaries"
