#!/bin/bash

set -eu

NUMA_NODE="0"
# NUMACTL_CORES="$((NUMA_NODE * 24))-$(((NUMA_NODE + 1) * 24 - 1))"
NUMACTL_CORES="$((NUMA_NODE * 24))-$(((NUMA_NODE + 1) * 24 - 1)),$((NUMA_NODE * 24 + 192))-$(((NUMA_NODE + 1) * 24 - 1 + 192))"

PROFILE_COMMAND=""
# PROFILE_COMMAND="perf stat -e instructions,cycles,cache-misses,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses --"
# PROFILE_COMMAND="AMDuProfPcm -m memory -a -d 60 -o /tmp/pcmdata.csv"

ENERGY0="$(cat "/sys/cray/pm_counters/energy" | awk '{ print $1 }')"
numactl -C "${NUMACTL_CORES}" -- /usr/bin/time --verbose -- ${PROFILE_COMMAND} ./tests/demo_collisions_integration
ENERGY1="$(cat "/sys/cray/pm_counters/energy" | awk '{ print $1 }')"
echo "Energy delta: $((ENERGY1 - ENERGY0))"
