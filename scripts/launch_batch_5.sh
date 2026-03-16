#!/bin/bash
# Launches a job with residues chosen according to the PBS job array index

index=$1
job_script=$2
labels=( [0]=S [1]=T [2]=Y [3]=ST [4]=STY )
residues=( [0]="['S']" [1]="['T']" [2]="['Y']" [3]="['S','T']" [4]="['S','T','Y']" )

echo "launching $job_script with index $index, residues ${residues[$index]}, label ${labels[$index]}"
"$job_script" "${labels[$index]}" "${residues[$index]}"
#echo "${residues[$index]}"
#echo "${labels[$index]}"
