!/bin/bash
#SBATCH --partition=gpu
# necessary to set the account also to the queue name because otherwise access is not allowed at the moment
##SBATCH --account=fwkt_v100
#SBATCH --time=4:00:00
# Sets batch job's name
#SBATCH --job-name=tests_detector
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=94500
#SBATCH --gres=gpu:1

#SBATCH -o stdout
#SBATCH -e stderr

source ~/gpu_picongpu.profile.detector
PYTEST_PATH=~/anaconda3/envs/detector_tests/bin/pytest # change it to your pytest path
PYTHONPATH="" $PYTEST_PATH ../../lib/python/picongpu/tests
