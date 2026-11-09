#!/bin/bash

set -eu

SCRIPT_DIRECTORY="$(dirname -- "$(realpath -- "${BASH_SOURCE[0]:-${0}}")")"

date
hostname
/usr/bin/rocm-smi --resetclocks
/usr/bin/rocm-smi

DEVICE_INDEX="4"
export ROCR_VISIBLE_DEVICES="${DEVICE_INDEX}"
export KOLIOP_DEBUG_koliop_Create="1"
export KOLIOP_DEBUG_PrepareBlockPosition="1"

PROFILE_COMMAND=""
# PROFILE_COMMAND="rocprofv3 --output-format=csv --stats --hip-trace --kernel-trace -- "
# PROFILE_COMMAND="rocprofv3 --output-format=pftrace --hip-runtime-trace --kernel-trace -- "

ENERGY0="$(cat "/sys/cray/pm_counters/accel$((DEVICE_INDEX / 2))_energy" | awk '{ print $1 }')"
numactl -C 0-7 -- /usr/bin/time --verbose -- ${PROFILE_COMMAND} ./tests/demo_collisions_integration
# numactl -C 0-7 -- /usr/bin/time --verbose -- ${PROFILE_COMMAND} ./tests/test_collisions_small_sanity --skip-f5d --write-reference "${SCRIPT_DIRECTORY}/references/references"
# numactl -C 0-7 -- /usr/bin/time --verbose -- ${PROFILE_COMMAND} ./tests/test_collisions_small_sanity --skip-f5d --reference "${SCRIPT_DIRECTORY}/references/references"
ENERGY1="$(cat "/sys/cray/pm_counters/accel$((DEVICE_INDEX / 2))_energy" | awk '{ print $1 }')"
echo "Energy delta: $((ENERGY1 - ENERGY0))"
