cwlVersion: v1.2
class: Workflow
label: "PIConGPU Simulation Workflow"
doc: |
  This workflow compiles and runs a PIConGPU simulation.
  First, it builds the PIConGPU executable using pic-build.
  Then, it runs the simulation using tbg.

inputs:
  build_include_directory:
    type: Directory
    label: "Compile-time parameter header directory"
    doc: "Directory containing compile-time parameter headers for compilation of PIConGPU"
  build_script:
    type: File
    label: "Build script"
    doc: "Shell script setting up the environment and running pic-build"
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
  run_etc_directory:
    type: Directory
    label: "Compile-time parameter header directory"
    doc: "Directory containing compile-time parameter headers for compilation of PIConGPU"
  prepare_submission_script:
    type: File
    label: "Prepare-submission script"
    doc: "Shell script setting up the environment for submission of the job"
  submission_script:
    type: File
    label: "Submission script"
    doc: "Shell script for submitting the prepared job"
  organize_output_script:
    type: File
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
    type: Directory
    label: "Setup path"
    doc: "Directory with the simulation setup to run"

outputs:
  input_directory:
    type: Directory
    outputSource: organize_output_step/input_directory
    label: "Input directory"
    doc: "Directory containing the original input as generated from Python"
  tbg_directory:
    type: Directory
    outputSource: organize_output_step/tbg_directory
    label: "Input directory"
    doc: "Directory containing the original input as generated from Python"
  submission_information:
    type: File
    outputSource: organize_output_step/submission_information
    label: "Submission information"
    doc: "This file contains sufficient information to manage the submitted job. Which precisely, depends on the submit_system."
  link_results_script:
    type: File
    outputSource: organize_output_step/link_results_script

steps:
  build_step:
    run: steps/build.cwl
    in:
      include_directory: build_include_directory
      script: build_script
      jobs: build_jobs
      cmake: build_cmake
      preset: build_preset
      force: build_force
      cmake_build_system: build_cmake_build_system
      help: build_help
    out: [bin_directory]
    label: "Build PIConGPU"
  prepare_submission_step:
    run: steps/prepare_submission.cwl
    in:
      etc_directory: run_etc_directory
      script: prepare_submission_script
      cfg_file: run_cfg_file
      overwrite_vars: run_overwrite_vars
      template_file: run_template_file
      force: run_force
    out: [tbg_directory]
  submit_step:
    run: steps/submit.cwl
    in:
      bin_directory: build_step/bin_directory
      etc_directory: run_etc_directory
      tbg_link: prepare_submission_step/tbg_directory
      script: submission_script
      submit_system: run_submit_system
    out: [submission_information, link_results_script, tbg_directory]
    label: "Submit PIConGPU simulation to the batch system"
  organize_output_step:
    run: steps/organize_output.cwl
    in:
      script: organize_output_script
      project_path: run_project_path
      bin_directory: build_step/bin_directory
      tbg_directory: submit_step/tbg_directory
      submission_information: submit_step/submission_information
      link_results_script: submit_step/link_results_script
    out: [input_directory, tbg_directory, submission_information, link_results_script]
