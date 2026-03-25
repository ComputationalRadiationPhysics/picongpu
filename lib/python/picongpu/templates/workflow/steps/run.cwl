cwlVersion: v1.2
class: CommandLineTool
label: "Run PIConGPU Simulation"
doc: "Run the compiled PIConGPU simulation using tbg"
baseCommand: bash
inputs:
  script:
      type: File
      inputBinding:
        position: 1
      label: "Run script"
      doc: "Shell script containing tbg command and flags"
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
