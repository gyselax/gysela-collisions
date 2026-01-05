module purge

module load cmake

# Just set LD_LIBRARY_PATH, LIBRARY_PATH and PATH towards an AOCC install.
# #!/bin/bash
# SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
# export AOCC_ROOT="${SCRIPT_DIRECTORY}"
# export LD_LIBRARY_PATH="${AOCC_ROOT}/lib:${LD_LIBRARY_PATH}"
# export LIBRARY_PATH="${AOCC_ROOT}/lib:${LIBRARY_PATH}"
# export PATH="${AOCC_ROOT}/bin:${PATH}"
# export PATH="${AOCC_ROOT}/share/opt-viewer:${PATH}"
# "${AOCC_ROOT}/AOCC-prerequisites-check.sh"

source -- ~/usr/bin/aocc/aocc-compiler-5.0.0/environment.sh

module list
