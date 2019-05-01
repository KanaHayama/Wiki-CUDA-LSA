#!/usr/bin/bash
## auto run in master node

slave_job_name_prefix="spark-worker-"
flag_filename="spark_cluster_running"
num_worker=$1

# cancel jobs
squeue -v | grep spark
for i in $(seq 1 $num_worker) ; do
	echo "Shutdown worker $i"
	scancel --name=${slave_job_name_prefix}$i
done
echo "Shutdown Spark cluster done!"
squeue -v | grep spark

# remove flag file
rm -f  $work_dir/$flag_filename