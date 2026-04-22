cwlVersion: v1.2
class: CommandLineTool
label: "Run PIConGPU Simulation"
doc: "Run the compiled PIConGPU simulation using tbg"

requirements:
  InitialWorkDirRequirement:
    listing:
      - entryname: etc
        entry: $(inputs.etc_directory)
      - entryname: prepare_submission.sh
        entry: $(inputs.script)
      - entryname: include
        entry: $(inputs.include_directory)
      - entryname: bin
        entry: $(inputs.bin_directory)

baseCommand: ./prepare_submission.sh

inputs:
  include_directory:
    type: Directory
    label: "Compile-time parameter header directory"
    doc: "Directory containing compile-time parameter headers for compilation of PIConGPU"
  etc_directory:
    type: Directory
    label: "Run-time configuration files for PIConGPU"
    doc: "Directory containing the run-time configuration files for PIConGPU"
  script:
    type: File
    label: "Prepare-submission script"
    doc: "Shell script setting up the environment for submission of the job"
  template_file:
    type: string?
    inputBinding:
      position: 1
      prefix: "-t"
    label: "TBG template file"
    doc: "Template to create a batch file from."
    default: ""
  cfg_file:
    type: string
    inputBinding:
      position: 2
      prefix: "-c"
    label: "Configuration file"
    doc: "Configuration file to set up batch file"
    default: "etc/picongpu/N.cfg"
  overwrite_vars:
    type: string?
    inputBinding:
      position: 5
      prefix: "-o"
    label: "Overwrite variables"
    doc: "Overwrite any template variable (JSON format)"
    default: null
  force:
    type: boolean
    inputBinding:
      position: 6
      prefix: "-f"
    label: "Force overwrite"
    doc: "Override if destinationPath exists"
    default: false
  help:
    type: boolean
    inputBinding:
      position: 7
      prefix: "-h"
    label: "Show help"
    doc: "Show the help message and exit"
    default: false
  bin_directory:
    type: Directory
    label: "PIConGPU executables"
    doc: "Compiled PIConGPU binaries from build step"
outputs:
  input_directory:
    type: Directory
    outputBinding:
      glob: "run_dir/input"
  tbg_directory:
    type: Directory
    outputBinding:
      glob: "run_dir/tbg"
