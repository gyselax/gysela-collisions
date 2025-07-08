#!/bin/bash

set -eu

NUMA_NODE="0"
# NUMACTL_CORES="$((NUMA_NODE * 24))-$(((NUMA_NODE + 1) * 24 - 1))"
NUMACTL_CORES="$((NUMA_NODE * 24))-$(((NUMA_NODE + 1) * 24 - 1)),$((NUMA_NODE * 24 + 192))-$(((NUMA_NODE + 1) * 24 - 1 + 192))"

ENERGY0="$(cat "/sys/cray/pm_counters/energy" | awk '{ print $1 }')"
/usr/bin/time --verbose numactl -C "${NUMACTL_CORES}" -- ./tests/demo_dummy_integration
ENERGY1="$(cat "/sys/cray/pm_counters/energy" | awk '{ print $1 }')"
echo "Energy delta: $((ENERGY1 - ENERGY0))"
