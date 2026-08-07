cd /storage/brno12-cerit/home/nexuso1/phossil/model
../scripts/pip_install.sh
residues=$1
splits=$2
prot_info=$3
type=$4
suffix=$5

OUT_DIRNAME=baseline_"$residues"_"$type"_"$suffix"

echo "Running experiment $OUT_DIRNAME with residues $residues"

if compgen -G "new_logs/$OUT_DIRNAME/fold_0/*.ckpt" > /dev/null; then
        echo "Log already exists, resuming from checkpoint"
        regex="fold_([0-9]+)"
        max=0
        final_path=""
        for path in new_logs/"$OUT_DIRNAME"/*/chkpt.ckpt;
        do
                dir="$(dirname $path)"
                dir="$(basename $dir)"
                if [[ "$dir" =~ $regex ]]
                then
                        num="${BASH_REMATCH[1]}"
                        max=$(( "$num" > "$max" ? "$num" : "$max" ))
                        final_path="$path"
                fi
        done
        echo "Resuming from $final_path"
        python3 baseline.py --checkpoint_path="$final_path"
else
	python3 baseline.py --epochs=30 --batch_size=4 --accum=8 --pos_weight=1 --prot_info_path "$prot_info" --dataset_path "$splits" --lr=5e-4 -o "$OUT_DIRNAME" --num_workers=15 --residues="$residues" --modify_prob=0 --type="$type"
fi
