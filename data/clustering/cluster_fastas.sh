#!/bin/bash

input_path=$1
output_path=$2


mmseqs createdb "$input_path" seqDB
#mmseqs2 makepaddedseqdb seqDB seqDB_gpu
# cov-mode 5 = shorter seq needs to be at least c% of the other seq
# alignment mode 3 = also compute seq. id.
# cluster mode 2 = greedy incremental
#
cluster_args="--min-seq-id 0.3 -c 0 --cov-mode 1 --cluster-mode 1 --alignment-mode 3 -s 6 --single-step-clustering"
cluster_args_path="${output_path%.*}"_cluster_args.txt
mmseqs cluster seqDB seqDB_clu tmp $cluster_args
echo "$cluster_args" > "$cluster_args_path"
mmseqs createtsv seqDB seqDB seqDB_clu "$output_path"
rm seqDB_clu.*