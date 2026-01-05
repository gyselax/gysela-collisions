#!/usr/bin/env python3

import sys
import csv
from collections import defaultdict


def BuildTraceHistogram(trace_lines):
    source_line_to_hit_count = defaultdict(int)
    trace_line_count = 0
    for itrace, a_trace_line in enumerate(trace_lines):
        if itrace == 0:
            # Skip the header
            continue
        trace_line_count += 1
        source_line_to_hit_count[a_trace_line[4]] += 1

        if trace_line_count % 400000 == 0 and trace_line_count != 0:
            print(f"Processed {trace_line_count} lines.", flush=True)

    return trace_line_count, source_line_to_hit_count


def ExcludeSourceLine(source_line):
    if not source_line:
        return True
    if "Unrecognized code object id" in source_line:
        return True
    # TODO(Etienne M): More exclusion.
    return False


def Present(source_line_histogram):
    trace_line_count, source_line_to_hit_count = source_line_histogram

    excluded_hit_count = 0
    hit_percentage_sum = 0.0

    def OrderingKey(source_line):
        source_line_splitted = source_line.split(":")
        if len(source_line_splitted) != 2:
            print(f"Weird source file {source_line_splitted}")
            return ("", 0)
        return (source_line_splitted[0], int(source_line_splitted[1]))

    source_file_to_hit_count = defaultdict(float)

    print("Per source line:")

    for source_line in sorted(source_line_to_hit_count, key=OrderingKey):
        source_line_hit_count = source_line_to_hit_count[source_line]
        if ExcludeSourceLine(source_line):
            excluded_hit_count += source_line_hit_count
            continue

        hit_percentage = (source_line_hit_count / trace_line_count) * 100.0
        hit_percentage_sum += hit_percentage

        source_file_to_hit_count[source_line.split(":")[0]] += hit_percentage

        print(f"'{source_line}' -> {hit_percentage:.2F}")
    print(f"excluded_hit_count: {excluded_hit_count}")
    print(f"hit_percentage_sum: {hit_percentage_sum:.2F}%")

    print("Per source file:")

    for source_file, hit_percentage in sorted(
        source_file_to_hit_count.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"'{source_file}' -> {hit_percentage:.2F}")


def ProcessCSVTrace(csv_trace_file_fullpath):
    with open(csv_trace_file_fullpath) as csv_trace:
        trace_lines = csv.reader(csv_trace, delimiter=",", quotechar='"')
        source_line_histogram = BuildTraceHistogram(trace_lines)
        Present(source_line_histogram)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print(
            "Collect traces like so:\n"
            "rocprofv3 --output-format=csv --pc-sampling-beta-enabled --pc-sampling-method host_trap --pc-sampling-unit time --pc-sampling-interval 1 -- ./my_program <arguments>"
        )
    ProcessCSVTrace(sys.argv[1])
