cwlVersion: v1.2
class: Workflow
label: "PIConGPU Simulation Workflow"
doc: |
  This workflow compiles and runs a PIConGPU simulation.
  First, it builds the PIConGPU executable using pic-build.
  Then, it runs the simulation using tbg.

inputs:
  build_script:
    type: File
    label: "Build script"
    doc: "Shell script to compile PIConGPU with pic-build"
  run_script:
    type: File
    label: "Run script"
    doc: "Shell script to run the simulation with tbg"

outputs:
  executables:
    type:
      type: array
      items: File
    outputSource: build_step/executables
    label: "Compiled PIConGPU executables"
  simulation_results:
    type: Directory
    outputSource: run_step/simulation_results
    label: "Simulation output directory"

steps:
  build_step:
    run: steps/build.cwl
    in:
      script: build_script
    out: [executables]
    label: "Build PIConGPU"
  run_step:
    run: steps/run.cwl
    in:
      script: run_script
      executables: build_step/executables
    out: [simulation_results]
    label: "Run PIConGPU simulation"
