cwlVersion: v1.2
class: CommandLineTool
label: "Organize output"
doc: |
  Assembles the input directory (the generated setup plus the compiled
  binaries) and collects the run artifacts (tbg, submission information,
  link_results script). The setup is generated into ``<run_dir>/input``
  up-front, so it is staged directly as ``input`` here rather than copied
  again.

requirements:
  InitialWorkDirRequirement:
    listing:
      - entryname: organize_output.sh
        entry: $(inputs.script)
      - entryname: input
        entry: $(inputs.project_path)
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
