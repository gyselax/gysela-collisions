#!/bin/bash

set -eu

/usr/bin/rocm-smi --resetclocks
/usr/bin/rocm-smi

DEVICE_INDEX="$((1 * 1))"
export ROCR_VISIBLE_DEVICES="${DEVICE_INDEX}"

ENERGY0="$(cat "/sys/cray/pm_counters/accel$((DEVICE_INDEX / 1))_energy" | awk '{ print $1 }')"
/usr/bin/time --verbose ./tests/demo_dummy_integration
ENERGY1="$(cat "/sys/cray/pm_counters/accel$((DEVICE_INDEX / 1))_energy" | awk '{ print $1 }')"
echo "Energy delta: $((ENERGY1 - ENERGY0))"
