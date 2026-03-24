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
outputs:
  executables:
    type:
      type: array
      items: File
    outputBinding:
      glob: "bin/*"
    label: "Compiled executables"
    doc: "Compiled PIConGPU binaries"
