login in HPC

all sh scripts execute permission
chmod +x *.sh

check hard coded parameters
parameters of cluster in master_start.sh
parameters of LSA in master_submit.sh

run prepare_environment.sh

run master_start.sh

run master_submit.sh {num-cores-per-worker} {num-total-mem-per-worker}

wait task finish

rm -f spark_cluster_running
OR
run master_stop.sh {num-of-workers}

