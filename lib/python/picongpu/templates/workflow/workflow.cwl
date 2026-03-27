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
  build_jobs:
    type: int?
    label: "Number of parallel jobs"
    doc: "Allow N jobs at once; infinite jobs if set to null"
    default: 4
  build_cmake:
    type: string?
    label: "Extra CMake arguments"
    doc: "Extra arguments passed straight to CMake"
    default: null
  build_preset:
    type: int?
    label: "CMake preset number"
    doc: "Configure this preset number from CMake flags"
    default: null
  build_force:
    type: boolean
    label: "Force rebuild"
    doc: "Clear CMake cache and force scan for new .param files"
    default: false
  build_cmake_build_system:
    type: string?
    label: "CMake build system"
    doc: "Select the build system used by CMake (e.g. Ninja)"
    default: null
  build_help:
    type: boolean
    label: "Show help"
    doc: "Show the help message and exit"
    default: false
  run_script:
    type: File
    label: "Run script"
    doc: "Shell script to run the simulation with tbg"
  run_cfg_file:
    type: string
    label: "Configuration file"
    doc: "Configuration file to set up batch file"
    default: "etc/picongpu/N.cfg"
  run_submit_system:
    type: string?
    label: "Submit system"
    doc: "Submit command (qsub, qsub -h, sbatch, ...)"
    default: "bash"
  run_template_file:
    type: string?
    label: "TBG template file"
    doc: "Template to create a batch file from."
    default: null
  run_overwrite_vars:
    type: string?
    label: "Overwrite variables"
    doc: "Overwrite any template variable (JSON format)"
    default: null
  run_force:
    type: boolean
    label: "Force overwrite"
    doc: "Override if destinationPath exists"
    default: false
  run_help:
    type: boolean
    label: "Show help"
    doc: "Show the help message and exit"
    default: false
  run_project_path:
    type: string
    label: "Setup path"
    doc: "Directory with the simulation setup to run"
  run_destination_path:
    type: string
    label: "Destination path"
    doc: "Output directory for simulation results"

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
      jobs: build_jobs
      cmake: build_cmake
      preset: build_preset
      force: build_force
      cmake_build_system: build_cmake_build_system
      help: build_help
    out: [executables]
    label: "Build PIConGPU"
  run_step:
    run: steps/run.cwl
    in:
      script: run_script
      cfg_file: run_cfg_file
      submit_system: run_submit_system
      overwrite_vars: run_overwrite_vars
      template_file: run_template_file
      force: run_force
      help: run_help
      project_path: run_project_path
      destination_path: run_destination_path
      executables: build_step/executables
    out: [simulation_results]
    label: "Run PIConGPU simulation"
