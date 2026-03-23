cwlVersion: v1.2
class: CommandLineTool
baseCommand: bash
inputs:
  script:
      type: File
      inputBinding:
        position: 1
  executables:
    type:
      type: array
      items: File
outputs:
  simulation_results:
    type:
      type: array
      items: File
    outputBinding:
      glob: "**/*.h5"
