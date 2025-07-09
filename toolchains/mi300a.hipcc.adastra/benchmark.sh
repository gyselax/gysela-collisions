#!/bin/bash

set -eu

/usr/bin/rocm-smi --resetclocks
/usr/bin/rocm-smi

DEVICE_INDEX="0"
export ROCR_VISIBLE_DEVICES="${DEVICE_INDEX}"

PROFILE_COMMAND=""
# PROFILE_COMMAND="rocprofv3 --output-format=csv --kernel-trace -- "
# PROFILE_COMMAND="rocprof --stats --timestamp on"

ENERGY0="$(cat "/sys/cray/pm_counters/accel$((DEVICE_INDEX / 1))_energy" | awk '{ print $1 }')"
numactl -C 0-23 -- /usr/bin/time --verbose -- ${PROFILE_COMMAND} ./tests/demo_dummy_integration
ENERGY1="$(cat "/sys/cray/pm_counters/accel$((DEVICE_INDEX / 1))_energy" | awk '{ print $1 }')"
echo "Energy delta: $((ENERGY1 - ENERGY0))"
