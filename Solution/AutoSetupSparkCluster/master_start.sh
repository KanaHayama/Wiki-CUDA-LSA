#!/usr/bin/bash
## manually run in master node

# init
work_dir="/staging/xq/zongjian/"
slave_job_name_prefix="spark-worker-"
num_worker=16
allocate_time="23:59:59"
allocate_cpu_mem=$SLURM_MEM_PER_CPU
allocate_core=$SLURM_CPUS_PER_TASK
flag_filename="spark_cluster_running"
master_host=$(hostname -f)
master_port=7077
master_url="spark://$master_host:$master_port"
# job_id=$(env  | grep SLURM_JOBID)
# master_ip=$(ifconfig | grep -E -o "inet (10\.[0-9\.]+)" | grep -E -o "[0-9\.]+")

# status
echo "Summary: $num_worker workers, each worker has $SLURM_CPUS_PER_TASK cpus, each worker has $allocate_cpu_mem M mem"

# start master node
current_dir=$(pwd)
cd $work_dir/spark/sbin
. "./stop-master.sh"
rm -f ../logs/*
./start-master.sh --host $master_host --port $master_port
echo "Master URL: $master_url"

# start slaves
cd $work_dir
touch $flag_filename
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
while [ $num_log -ne $(expr $num_worker + 1) ]; do
	sleep 10
	num_log=$(ls -l $work_dir/spark/logs |grep "^-"|wc -l)
	echo "Check worker state: $(expr $num_log - 1) worker(s) are ready"
done

# finish
echo "Setup Spark cluster done!"

# do computation here
./master_work.sh $master_url $(expr $num_worker + 1) $allocate_core $(expr $allocate_core \* $allocate_cpu_mem)

# stop workers
cd $work_dir
rm -f $flag_filename
cd $current_dir
./master_stop.sh $slave_job_name_prefix $num_worker

