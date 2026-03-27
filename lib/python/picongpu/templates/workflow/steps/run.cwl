cwlVersion: v1.2
class: CommandLineTool
label: "Run PIConGPU Simulation"
doc: "Run the compiled PIConGPU simulation using tbg"
hints:
  SoftwareRequirement:
    packages:
      picongpu:
        specs: ["https://doi.org/10.5281/zenodo.14513363"]
baseCommand: bash
inputs:
  script:
      type: File
      inputBinding:
        position: 1
      label: "Run script"
      doc: "Shell script containing tbg command and flags"
  cfg_file:
    type: string
    inputBinding:
      position: 2
      prefix: "-c"
    label: "Configuration file"
    doc: "Configuration file to set up batch file"
    default: "etc/picongpu/N.cfg"
  submit_system:
    type: string?
    inputBinding:
      position: 3
      prefix: "-s"
    label: "Submit system"
    doc: "Submit command (qsub, qsub -h, sbatch, ...)"
    default: "bash"
  template_file:
    type: string?
    inputBinding:
      position: 4
      prefix: "-t"
    label: "TBG template file"
    doc: "Template to create a batch file from."
    default: null
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
  project_path:
    type: string
    inputBinding:
      position: 8
    label: "Setup path"
    doc: "Directory with the simulation setup to run"
  destination_path:
    type: string
    inputBinding:
      position: 9
    label: "Destination path"
    doc: "Output directory for simulation results"
  executables:
    type:
      type: array
      items: File
    label: "PIConGPU executables"
    doc: "Compiled PIConGPU binaries from build step"
outputs:
  simulation_results:
    type: Directory
    label: "Simulation results"
    doc: "Output files from the simulation (HDF5 format)"
    outputBinding:
      glob: "results"
