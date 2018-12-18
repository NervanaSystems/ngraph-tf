export NNP_B_STEP=1
ARGON_API_LOG_ENABLE=1 NGRAPH_TF_BACKEND=NNP NGRAPH_TF_DUMP_GRAPHS=1  mpirun -np 2  --oversubscribe -x LD_LIBRARY_PATH -H localhost   python ./mnist_softmax_distributed.py --num_inter_threads 1
