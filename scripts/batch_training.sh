#!/bin/bash
# Launches a job with residues chosen according to the PBS job array index.
#
# Arguments are named and may be given in any order, either as "--arg value" or
# "--arg=value". Anything that is not one of the arguments below is passed
# through to the job script, after the positional arguments it already expects.
#
#   --index          PBS job array index, selects the residue set (required)
#   --job_script     script to launch (required)
#   --dataset        dbptm | phos | deeppsp (required)
#   --splits_suffix  suffix of the splits file, e.g. "_max" (default: empty)
#   --type           model type passed to the job script
#   --data_dir       root of the data directory (default: <script dir>/../data)
#   --name           run name prefix, enables resuming (see below)
#   --log_dir        directory holding the run directories
#                    (default: <script dir>/../model/new_logs)
#
# If --name is given, the run directory is <name>_<residues>_<type>_<suffix>,
# the same name the job scripts build. When that directory already holds a
# checkpoint, the job script is launched as
#
#   <job_script> --checkpoint_path=<latest fold checkpoint>
#
# and all other arguments are dropped -- the training args are restored from the
# run's metadata.json. Without --name no checkpoint lookup happens and the job
# script is always launched with the full argument list.

set -u

usage() {
    sed -n '2,27p' "$0"
    exit "${1:-1}"
}

index=""
job_script=""
dataset=""
splits_suffix=""
type=""
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/../data"
name=""
log_dir="$script_dir/../model/new_logs"

# Arguments that are not recognized below, forwarded to the job script as-is
extra_args=()

while [[ $# -gt 0 ]]; do
    # Split "--arg=value" into an argument and a value, leave "--arg value" alone
    case "$1" in
        --*=*)
            arg="${1%%=*}"
            value="${1#*=}"
            has_inline_value=1
            ;;
        *)
            arg="$1"
            value="${2-}"
            has_inline_value=0
            ;;
    esac

    known=1
    case "$arg" in
        --index) index="$value" ;;
        --job_script) job_script="$value" ;;
        --dataset) dataset="$value" ;;
        --splits_suffix) splits_suffix="$value" ;;
        --type) type="$value" ;;
        --data_dir) data_dir="$value" ;;
        --name) name="$value" ;;
        --log_dir) log_dir="$value" ;;
        -h|--help) usage 0 ;;
        *) known=0 ;;
    esac

    if [[ "$known" -eq 0 ]]; then
        extra_args+=("$1")
        shift
        continue
    fi

    if [[ "$has_inline_value" -eq 1 ]]; then
        shift
    else
        if [[ $# -lt 2 ]]; then
            echo "Missing value for $arg" >&2
            exit 1
        fi
        shift 2
    fi
done

residues=( [0]=S [1]=T [2]=Y [3]=ST [4]=STY )

if [[ -z "$index" || -z "$job_script" || -z "$dataset" ]]; then
    echo "--index, --job_script and --dataset are required" >&2
    usage
fi

if [[ ! "$index" =~ ^[0-9]+$ ]] || [[ -z "${residues[$index]+set}" ]]; then
    echo "Invalid index '$index', has to be one of: ${!residues[*]}" >&2
    exit 1
fi

declare -A prot_info
prot_info=( ["dbptm"]="$data_dir"/dbptm/dbptm_info_chunked.json ["phos"]="$data_dir"/phosphosite_sequences/phosphosite_df_chunked.json ["deeppsp"]="$data_dir"/deeppsp/dpsp_info_"${residues[$index]}"_chunked.json )
declare -A suffix
suffix=( ["dbptm"]=dbptm"$splits_suffix" ["phos"]=phos"$splits_suffix" ["deeppsp"]=dpsp"$splits_suffix" )

if [[ -z "${prot_info[$dataset]+set}" ]]; then
    echo "Unknown dataset '$dataset', has to be one of: ${!prot_info[*]}" >&2
    exit 1
fi

splits_path="$data_dir"/"$dataset"/splits_"${residues[$index]}""$splits_suffix".json

# Latest checkpoint of the run, empty if the run has not been started yet
find_checkpoint() {
    local out_dirname="$1"
    local regex="fold_([0-9]+)"
    local max=-1
    local final_path=""
    local path dir num

    if ! compgen -G "$log_dir/$out_dirname/fold_0/*.ckpt" > /dev/null; then
        return
    fi

    for path in "$log_dir/$out_dirname"/*/chkpt.ckpt; do
        [[ -f "$path" ]] || continue
        dir="$(basename "$(dirname "$path")")"
        if [[ "$dir" =~ $regex ]]; then
            num="${BASH_REMATCH[1]}"
            # Glob order is lexicographic, so fold_10 sorts before fold_2
            if (( num > max )); then
                max="$num"
                final_path="$path"
            fi
        fi
    done

    echo "$final_path"
}

checkpoint_path=""
if [[ -n "$name" ]]; then
    out_dirname="$name"_"${residues[$index]}"_"$type"_"${suffix[$dataset]}"
    checkpoint_path="$(find_checkpoint "$out_dirname")"
    echo "Run directory: $log_dir/$out_dirname"
fi

if [[ -n "$checkpoint_path" ]]; then
    echo "Log already exists, resuming from checkpoint $checkpoint_path"
    "$job_script" --checkpoint_path="$checkpoint_path"
else
    echo "launching $job_script with index $index, residues ${residues[$index]}, splits path: $splits_path prot info: ${prot_info[$dataset]} suffix ${suffix[$dataset]}"
    "$job_script" "${residues[$index]}" "$splits_path" "${prot_info[$dataset]}" "$type" "${suffix[$dataset]}" ${extra_args[@]+"${extra_args[@]}"}
fi
