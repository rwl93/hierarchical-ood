#!/bin/bash
#####################################################################
# Script to make symlinks to data for a given set of input labels   #
#####################################################################

USAGE="Generate symlinks to ImageNet data for a set of labels

Usage: gen-symlinks <label-file.csv> <data-dir> <dest-dir> <granularity-filter>
"

if [[ $# -eq 3 ]]
then
  filter="^."
elif [[ $# -eq 4 ]]
then
  filter=$4
else
  echo "$USAGE"
  exit 1
fi

echo "Filter by: $filter"
while IFS= read -r LINE; do
  echo "line: $LINE"
  label=$(sed -e "/$filter/!s/.*/NOMATCH/g" -e "s/,.*//g" <<< "$LINE")
  if [[ "$label" != "NOMATCH" ]]
  then
    echo "EXECUTING: ln -s $2/$label/ $3/"
    ln -s $2/$label/ $3/
  fi
done < $1
