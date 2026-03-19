cwlVersion: v1.2
class: Workflow

outputs:
  executables:
    type: array
    items: File
    outputBinding:
      glob: "bin/*"
  simulation_results:
    type: array
    items: File

steps:
  build_step:
    run: commands/build.sh
    out: [executables]
  run_step:
    run: commands/run.sh
    in: [executables]
    out: [simulation_results]
