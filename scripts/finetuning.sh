#cd /storage/brno12-cerit/home/nexuso1/phossil/model
#../scripts/pip_install.sh
cd ../model

residues=$1
splits=$2
prot_info=$3
type=$4
suffix=$5

OUT_DIRNAME=finetuning_"$residues"_3_"$type"_"$suffix"

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
        python3 ft_selective.py --checkpoint_path="$final_path"
else
	python3 ft_selective.py --epochs=30 --batch_size=16 --accum=2 --pos_weight=3 --prot_info_path "$prot_info" --dataset_path "$splits" --lr=5e-5 -o "$OUT_DIRNAME" --num_workers=15 --residues="$residues" --modify_prob=0 --type="$type" --indices="[-1, -2, -3]"
fi
