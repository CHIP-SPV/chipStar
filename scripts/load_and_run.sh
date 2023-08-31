#!/bin/bash

# Check if there are at least three arguments
if [ $# -lt 3 ]; then
  echo "Usage: $0 <rgx> <NUM_TRIES> <NUM_THREADS> [modules...]"
  exit 1
fi

# Check if 2nd and 3rd arguments are integers
if ! [[ "$2" =~ ^[0-9]+$ ]] || ! [[ "$3" =~ ^[0-9]+$ ]]; then
  echo "2nd and 3rd arguments must be integers"
  exit 1
fi

# Save the first argument to variable rgx
rgx="$1"
# Save the second argument to variable NUM_TRIES
NUM_TRIES="$2"
# Save the third argument to variable NUM_THREADS
NUM_THREADS="$3"

# Shift to get rid of the first three arguments
shift 3

# Pass all remaining arguments to "module load"
module load "$@"

ctest -R "$rgx" -j "$NUM_THREADS" --repeat until-fail:"$NUM_TRIES" --timeout 180