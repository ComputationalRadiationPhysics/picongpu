cwlVersion: v1.2
class: CommandLineTool
label: "Run PIConGPU Simulation"
doc: "Run the compiled PIConGPU simulation using tbg"

requirements:
  InitialWorkDirRequirement:
    listing:
      - entryname: submit.sh
        entry: $(inputs.script)
      - entryname: input/bin
        entry: $(inputs.bin_directory)
      - entryname: tbg
        entry: $(inputs.tbg_directory)

baseCommand: ./submit.sh

inputs:
  script:
    type: File
    label: "Submission script"
    doc: "Shell script for submitting the prepared job"
  bin_directory:
    type: Directory
  tbg_directory:
    type: Directory
  submit_system:
    type: string?
    inputBinding:
      position: 2
    label: "Submit system"
    doc: "Submit command (qsub, sbatch, ...)"
    default: "bash"
outputs:
  submission_information:
    type: File
    label: "Submission information file"
    doc: "This file contains sufficient information to manage the submitted job. Which precisely, depends on the submit_system."
    outputBinding:
      glob: "submission_information.txt"
