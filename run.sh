for size in 64;
do
	for numlayers in 3 2 1;
	do
		tmux split-window "source activate deep; \
		    python classification.py $size $numlayers 0.25 --gpu_frac 0.31 --gpu_id 0 --model_type gru; \
		    read"
		sleep 15
	done
done
