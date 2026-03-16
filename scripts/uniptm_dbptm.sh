cd /storage/brno12-cerit/home/nexuso1/phossil/model
../scripts/pip_install.sh

label=$1
residues=$2

OUT_DIRNAME=uniptm_"$label"_dbptm
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
	python3 uniptm.py --checkpoint_path="$final_path"
else
	python3 uniptm.py --epochs=300 --batch_size=8 --accum=4 --pos_weight=3 --dataset_path "../data/dbptm/splits_$label.json" --lr=5e-5 --dropout=0.5 -o "$OUT_DIRNAME" --num_workers=15 --patience=40 --prot_info_path='../data/dbptm/dbptm_info.json' --residues=$residues --step_lr

fi
