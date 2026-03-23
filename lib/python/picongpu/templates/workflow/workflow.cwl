cwlVersion: v1.2
class: Workflow

inputs:
  build_script:
    type: File
  run_script:
    type: File

outputs:
  executables:
    type:
      type: array
      items: File
    outputSource: build_step/executables

steps:
  build_step:
    run: steps/build.cwl
    in:
      script: build_script
    out: [executables]
  run_step:
    run: steps/run.cwl
    in:
      script: run_script
      executables: build_step/executables
    out: [simulation_results]
