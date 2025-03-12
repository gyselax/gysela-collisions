module purge

module load cmake

# Just set LD_LIBRARY_PATH, LIBRARY_PATH and PATH towards an AOCC install.
source -- ~/usr/bin/aocc/aocc-compiler-5.0.0/environment.sh

module list
