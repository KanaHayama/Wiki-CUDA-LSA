#!/usr/bin/bash
## manually run in master node

# init
work_dir="/staging/xq/zongjian/"
slave_job_name_prefix="spark-worker-"
num_worker=64
allocate_time="23:59:59"
allocate_cpu_mem=$SLURM_MEM_PER_CPU
allocate_core=$SLURM_CPUS_PER_TASK
job_id=$(env  | grep SLURM_JOBID)
master_ip=$(ifconfig | grep -E -o "inet (10\.[0-9\.]+)" | grep -E -o "[0-9\.]+")

# status
echo "Summary: $num_worker workers, each worker has $SLURM_CPUS_PER_TASK cpus, each worker has $allocate_cpu_mem M mem"

# start master node
current_dir=$(pwd)
cd $work_dir/spark/sbin
. "./stop-master.sh"
rm -f ../logs/*
log_file=$(. "./start-master.sh" | grep -E -o "/.*out")
master_url=""
while [ "$master_url" == "" ]; do
	sleep 3
	master_url=$(cat $log_file | grep -E -o "spark:.*$")
	echo "Check master state"
done
echo "Master URL: $master_url"


# start slaves
cd $work_dir
if [ ! -d "logs" ]; then
	mkdir logs
fi
rm -f logs/*
cd $current_dir
for i in $(seq 1 $num_worker) ; do
	echo "Launch worker $i"
	sbatch -N1 -n1 --job-name=$slave_job_name_prefix$i --time=$allocate_time --cpus-per-task=$allocate_core --mem-per-cpu=$allocate_cpu_mem worker.sh $master_url
done

# check
num_log=1
while [ $num_log -ne $SLURM_NNODES ]; do
	sleep 10
	num_log=$(ls -l $work_dir/spark/logs |grep "^-"|wc -l)
	echo "Check worker state: $(expr $num_log - 1) worker(s) are ready"
done

# finish
echo "Setup Spark cluster done!"

# do computation here
cd $work_dir
# ./prepare_environment.sh
# echo "Environment: $HADOOP_HOME, $SPARK_HOME, $MAHOUT_HOME, $SCALA_HOME"

# stop workers
cd $current_dir
squeue -v | grep spark
for i in $(seq 1 $num_worker) ; do
	echo "Shutdown worker $i"
	scancel --name=$slave_job_name_prefix$i
done
echo "Shutdown Spark cluster done!"
squeue -v | grep spark