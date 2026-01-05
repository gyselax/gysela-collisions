module purge

module load cmake

module load cpe/24.07
module load craype-x86-genoa
module load PrgEnv-intel
module load intel-classic
module unload cray-libsci

module list
