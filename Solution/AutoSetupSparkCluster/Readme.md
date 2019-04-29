login in HPC

all sh scripts chmod +x

check parameters

run prepare_environment.sh

run allocate_node.sh

wait for a node assigned

run master_start.sh

wait task finish

if aborted {
	rm -f spark_cluster_running
	to stop workers
}

