cwlVersion: v1.2
class: CommandLineTool
label: "Organize output"
doc: "Sets up the expected directory structure particularly by copying over the input"

requirements:
  InitialWorkDirRequirement:
    listing:
      - entryname: organize_output.sh
        entry: $(inputs.script)
  EnvVarRequirement:
    envDef:
      - envName: PICONGPU_RUNNING_AS_CWL
        envValue: "1"

baseCommand: ./organize_output.sh

inputs:
  script:
    type: File
  project_path:
    type: Directory
    inputBinding:
      position: 1
  bin_directory:
    type: Directory
    inputBinding:
      position: 2
  tbg_directory:
    type: Directory
    inputBinding:
      position: 3
  submission_information:
    type: File
    inputBinding:
      position: 4
  link_results_script:
    type: File
    inputBinding:
      position: 5
outputs:
  input_directory:
    type: Directory
    outputBinding:
      glob: "input"
  tbg_directory:
    type: Directory
    outputBinding:
      glob: "tbg"
  link_results_script:
    type: File
    outputBinding:
      glob: "link_results.sh"
  submission_information:
    type: File
    outputBinding:
      glob: "submission_information.txt"
