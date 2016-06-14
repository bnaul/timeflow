#tmux set-option remain-on-exit on
for lstmsize in 128;
do
	for numlayers in 3 2 1;
	do
		tmux new-window "source activate deep; python period.py $lstmsize $numlayers 0.25; read"
		sleep 15
	done
done
