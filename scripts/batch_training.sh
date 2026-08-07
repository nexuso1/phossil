#!/bin/bash
# Launches a job with residues chosen according to the PBS job array index

index=$1
job_script=$2
dataset=$3
splits_suffix=$4
type=$5
data_dir=$6

echo "$dataset"

residues=( [0]=S [1]=T [2]=Y [3]=ST [4]=STY )
declare -A prot_info
prot_info=( ["dbptm"]="$data_dir"/dbptm/dbptm_info_chunked.json ["phos"]="$data_dir"/phosphosite_sequences/phosphosite_df_chunked.json ["deeppsp"]="$data_dir"/deeppsp/dpsp_info_"${residues[$index]}"_chunked.json )
declare -A suffix
suffix=( ["dbptm"]=dbptm"$splits_suffix" ["phos"]=phos"$splits_suffix" ["deeppsp"]=dpsp"$splits_suffix" )
splits_path="$data_dir"/"$dataset"/splits_"${residues[$index]}""$splits_suffix".json

echo "$dataset"
echo "${prot_info[${dataset}]}"

echo "launching $job_script with index $index, residues ${residues[$index]}, splits path: $splits_path prot info: ${prot_info[$dataset]} suffix ${suffix[$dataset]}"
"$job_script" "${residues[$index]}" "$splits_path" "${prot_info[$dataset]}" "$type" "${suffix[$dataset]}"

