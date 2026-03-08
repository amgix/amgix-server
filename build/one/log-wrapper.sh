#!/bin/bash
# Wrapper to prefix log lines with service name
# Usage: log-wrapper.sh <prefix> <command> [args...]

PREFIX="$1"
shift

# Run command and prefix each line of stdout and stderr
{
    "$@" 2>&1 | while IFS= read -r line; do
        echo "[$PREFIX] $line"
    done
} &

# Wait for the command to finish and preserve exit code
wait $!

