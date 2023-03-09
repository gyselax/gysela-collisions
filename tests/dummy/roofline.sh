#!/bin/bash

set -eu

# Sources:
# https://github.com/ROCm-Developer-Tools/rocprofiler/blob/amd-master/test/tool/metrics.xml
# https://github.com/ROCm-Developer-Tools/rocprofiler/blob/amd-master/test/tool/gfx_metrics.xml

# TCC_EA_RDREQ_sum
# TCC_EA_WRREQ_sum
# TCC_EA_RDREQ_32B_sum
# TCC_EA_WRREQ_64B_sum

# SQ_INSTS_VALU_ADD_F16
# SQ_INSTS_VALU_MUL_F16
# SQ_INSTS_VALU_FMA_F16
# SQ_INSTS_VALU_TRANS_F16
# SQ_INSTS_VALU_MFMA_MOPS_F16
# SQ_INSTS_VALU_MFMA_MOPS_BF16

# SQ_INSTS_VALU_ADD_F32
# SQ_INSTS_VALU_MUL_F32
# SQ_INSTS_VALU_FMA_F32
# SQ_INSTS_VALU_TRANS_F32
# SQ_INSTS_VALU_MFMA_MOPS_F32

# SQ_INSTS_VALU_ADD_F64
# SQ_INSTS_VALU_MUL_F64
# SQ_INSTS_VALU_FMA_F64
# SQ_INSTS_VALU_TRANS_F64
# SQ_INSTS_VALU_MFMA_MOPS_F64

# cat <<EOF > program_counters.txt
# pmc : TCC_EA_RDREQ_32B_sum TCC_EA_RDREQ_sum TCC_EA_WRREQ_sum TCC_EA_WRREQ_64B_sum SQ_INSTS_VALU_ADD_F16 SQ_INSTS_VALU_MUL_F16 SQ_INSTS_VALU_FMA_F16 SQ_INSTS_VALU_TRANS_F16 SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32 SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_TRANS_F32
# pmc : SQ_INSTS_VALU_ADD_F64 SQ_INSTS_VALU_MUL_F64 SQ_INSTS_VALU_FMA_F64 SQ_INSTS_VALU_TRANS_F64 SQ_INSTS_VALU_MFMA_MOPS_F16 SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_INSTS_VALU_MFMA_MOPS_F32 SQ_INSTS_VALU_MFMA_MOPS_F64
# gpu: 0
# kernel: void Kokkos::Experimental::Impl::hip_parallel_launch_local_memory<Kokkos::Impl::ParallelFor<koliop::collision::State<koliop::collision::CollisionPolicy>::DoMultiply<Kokkos::Experimental::HIP, Kokkos::View<double**, Kokkos::Experimental::HIPSpace>, Kokkos::View<double*, Kokkos::Experimental::HIPSpace>, Kokkos::View<double*, Kokkos::Experimental::HIPSpace> >(Kokkos::Experimental::HIP const&, Kokkos::View<double**, Kokkos::Experimental::HIPSpace> const&, Kokkos::View<double*, Kokkos::Experimental::HIPSpace> const&, Kokkos::View<double*, Kokkos::Experimental::HIPSpace>&)::{lambda(int)#1}, Kokkos::RangePolicy<Kokkos::Experimental::HIP>, Kokkos::Experimental::HIP>, 1024u, 1u>(Kokkos::Impl::ParallelFor<koliop::collision::State<koliop::collision::CollisionPolicy>::DoMultiply<Kokkos::Experimental::HIP, Kokkos::View<double**, Kokkos::Experimental::HIPSpace>, Kokkos::View<double*, Kokkos::Experimental::HIPSpace>, Kokkos::View<double*, Kokkos::Experimental::HIPSpace> >(Kokkos::Experimental::HIP const&, Kokkos::View<double**, Kokkos::Experimental::HIPSpace> const&, Kokkos::View<double*, Kokkos::Experimental::HIPSpace> const&, Kokkos::View<double*, Kokkos::Experimental::HIPSpace>&)::{lambda(int)#1}, Kokkos::RangePolicy<Kokkos::Experimental::HIP>, Kokkos::Experimental::HIP> const*) [clone .kd]
# EOF

function Binary64Counters() {
    local COUNTERS="TCC_EA_RDREQ_32B_sum TCC_EA_RDREQ_sum TCC_EA_WRREQ_sum TCC_EA_WRREQ_64B_sum SQ_INSTS_VALU_ADD_F64 SQ_INSTS_VALU_MUL_F64 SQ_INSTS_VALU_FMA_F64 SQ_INSTS_VALU_TRANS_F64 SQ_INSTS_VALU_MFMA_MOPS_F64"
    echo "pmc : ${COUNTERS}" >>program_counters.txt
}

function Binary32Counters() {
    local COUNTERS="TCC_EA_RDREQ_32B_sum TCC_EA_RDREQ_sum TCC_EA_WRREQ_sum TCC_EA_WRREQ_64B_sum SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32 SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_TRANS_F32 SQ_INSTS_VALU_MFMA_MOPS_F32"
    echo "pmc : ${COUNTERS}" >>program_counters.txt
}

function Binary16Counters() {
    local COUNTERS="TCC_EA_RDREQ_32B_sum TCC_EA_RDREQ_sum TCC_EA_WRREQ_sum TCC_EA_WRREQ_64B_sum SQ_INSTS_VALU_ADD_F16 SQ_INSTS_VALU_MUL_F16 SQ_INSTS_VALU_FMA_F16 SQ_INSTS_VALU_TRANS_F16 SQ_INSTS_VALU_MFMA_MOPS_F16 SQ_INSTS_VALU_MFMA_MOPS_BF16"
    echo "pmc : ${COUNTERS}" >>program_counters.txt
}

echo -n "" >program_counters.txt
echo "gpu: 0 " >>program_counters.txt
echo "kernel : " >>program_counters.txt

################################################################################

# rocprof will start the code as many times as we have specified a "pcm:"
# keyword in the input file.
# NOTE: it is wellknown that rocm 5.2.0 and earlier contains a rocprof version
# that returns faulty timestamps when "pcm:" is specified in the input. If one
# absolutely needs rocm 5.2.0, one can do one run with and one run without the
# "pcm:" to collect separately the counters values and the timestamps.

case "${3}" in
"64")
    echo "Gathering Binary64 metrics."
    echo
    Binary64Counters
    ;;
"32")
    echo "Gathering Binary32 metrics."
    echo
    Binary32Counters
    ;;
"16")
    echo "Gathering Binary16 metrics."
    echo
    Binary16Counters
    ;;
*)
    echo "Requires 5.3.0 and up ROCm version (if you want to avoid troubles."
    echo "Usage: script.sh <executable> <prefix> <flop category>"
    echo "We do not support giving arguments to the <executable>."
	echo "<prefix> can be used to differentiate the output when multiple"
    echo "process are started (say, using MPI)."
    echo "Where <flop category> is one of [64, 32, 16]. rocprof will gather"
    echo "Flop data from the kernel regarding only the Flop category you"
    echo "specified. Flop category meaning, say for 64, IEEE 754's Binary64."
    echo "Except that for Binary16, we also measure Brain16 Flop."
    exit -1
    ;;
esac

################################################################################

export AMD_LOG_LEVEL=1

PREFIX="${2}"
OUTPUT_FILENAME="${PREFIX}_metrics_$(date +%s).csv"

rocprof -i program_counters.txt --timestamp on -o "${OUTPUT_FILENAME}" "${1}"

tail -- "${OUTPUT_FILENAME}"
