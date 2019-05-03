+Login in HPC


+Adding all shell scripts execute permission

chmod +x *.sh


+Check hard coded parameters

parameters of Spark cluster in master_start.sh

parameters of LSA in master_submit.sh


+Setup libraries and dataset

run prepare_environment.sh


+Start Spark cluster using standalone cluster manager

run master_start.sh


+Submit LSA job

run master_submit.sh {num-cores-per-worker} {num-total-mem-per-worker}


+Wait task finish


+Stop Spark cluster

rm -f spark_cluster_running

OR

run master_stop.sh {num-of-workers}
