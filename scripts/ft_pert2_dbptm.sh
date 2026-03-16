cd /storage/brno12-cerit/home/nexuso1/phossil/model
../scripts/pip_install.sh
label=$1
residues=$2

OUT_DIRNAME=finetuning_"$label"_3_perturb2_dbptm

echo "Running experiment $OUT_DIRNAME with label $label and residues $residues"

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
	python3 ft_selective.py --epochs=30 --batch_size=4 --accum=8 --pos_weight=3 --prot_info_path ../data/dbptm/dbptm_info.json --dataset_path=../data/dbptm/splits_"$label".json --lr=5e-5 -o "$OUT_DIRNAME" --num_workers=15 --dropout=0.5 --indices="[-1, -2, -3]" --residues=$residues --hidden_size=256 --modify_prob=0.15 --rand_prob=0 --mask_prob=0 --sub_prob=1
fi
