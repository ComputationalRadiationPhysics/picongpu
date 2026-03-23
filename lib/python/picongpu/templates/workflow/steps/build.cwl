cwlVersion: v1.2
class: CommandLineTool
baseCommand: bash
inputs:
  script:
      type: File
      inputBinding:
        position: 1
outputs:
  executables:
    type:
      type: array
      items: File
    outputBinding:
      glob: "bin/*"
