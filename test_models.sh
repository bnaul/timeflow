set -ex

TEST_VALS="--n_min 10 --n_max 10 --nb_epoch 2 --N_train 10 --N_test 10"
rm -rf keras_logs/test*

for file in period period_inverse autoencoder;
do
    for num_layers in 1 2;
    do
        for even in even uneven;
        do
            python $file.py 4 $num_layers 0.25 --sim_type test --$even $TEST_VALS
            rm -rf keras_logs/test*
        done
    done
done
echo "All tests passed."
