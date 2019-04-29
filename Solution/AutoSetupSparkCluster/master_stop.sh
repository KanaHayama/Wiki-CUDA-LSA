#!/usr/bin/bash
## auto run in master node

slave_job_name_prefix=$1
num_worker=$2

squeue -v | grep spark
for i in $(seq 1 $num_worker) ; do
	echo "Shutdown worker $i"
	scancel --name=$slave_job_name_prefix$i
done
echo "Shutdown Spark cluster done!"
squeue -v | grep spark