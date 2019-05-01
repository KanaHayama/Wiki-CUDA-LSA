login in HPC

all sh scripts chmod +x

check hard coded parameters

run prepare_environment.sh

run allocate_node.sh

wait for a node assigned

run master_start.sh

run master_submit.sh

wait task finish

rm -f spark_cluster_running
OR
run master_stop.sh {num-of-workers}

