module purge

module load cpe/25.09
module load cmake ninja

CUDA_PATH="/lus/work/BCINES/dci/malaboeuf/Spackle/UserProducts/Binaries/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_/linux-x86_64_v3/cuda-12.9.1-ukofbb4akbr5boya726n7jfd6wlzt4xw"

export PATH="${CUDA_PATH}/libnvvp:${PATH}"
export PATH="${CUDA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_PATH}/nvvm/lib64:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_PATH}/extras/Debugger/lib64:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_PATH}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

module list
