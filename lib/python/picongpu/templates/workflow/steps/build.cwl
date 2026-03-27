cwlVersion: v1.2
class: CommandLineTool
label: "Build PIConGPU"
doc: "Compile PIConGPU using pic-build with the provided build script"
hints:
  SoftwareRequirement:
    packages:
      picongpu:
        package: "PIConGPU"
        specs: ["https://doi.org/10.5281/zenodo.14513363"]
baseCommand: bash
inputs:
  script:
      type: File
      inputBinding:
        position: 1
      label: "Build script"
      doc: "Shell script containing pic-build command and flags"
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
  help:
    type: boolean
    inputBinding:
      position: 7
      prefix: "-h"
    label: "Show help"
    doc: "Show the help message and exit"
    default: false
outputs:
  executables:
    type:
      type: array
      items: File
    outputBinding:
      glob: "bin/*"
    label: "Compiled executables"
    doc: "Compiled PIConGPU binaries"
