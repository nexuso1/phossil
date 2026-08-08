#!/bin/bash
# Runs a positive class weight sweep for the given residue set

residues=$1
splits=$2
prot_info=$3
type=$4
suffix=$5

script_dir="$(dirname "$0")"

# Bash has no nested arrays, so the weights of a residue set are one space separated string
declare -A weights
weights=( ["S"]="3 4 4.3" ["T"]="2 3 4 5 5.29" ["Y"]="2 3 4 5 5.33" ["ST"]="2 3 4 5.03" ["STY"]="2 3 4 5 5.44" )

if [[ -z "${weights[$residues]+set}" ]]; then
    echo "No weights defined for residue set '$residues'. Known sets: ${!weights[*]}" >&2
    exit 1
fi

# Unquoted on purpose, the string has to be split into the individual weights
for weight in ${weights[$residues]}; do
    "$script_dir/baseline_pw.sh" "$residues" "$splits" "$prot_info" "$type" "$suffix" "$weight"
done
