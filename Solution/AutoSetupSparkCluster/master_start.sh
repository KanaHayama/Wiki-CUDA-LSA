#!/usr/bin/bash
## manually run in master node

# init
num_worker=8
allocate_core=8
allocate_cpu_mem="4g"
allocate_time="23:59:59"
allocate_time="1:00:00"

work_dir="/staging/xq/zongjian/"
slave_job_name_prefix="spark-worker-"
flag_filename="spark_cluster_running"
master_host=$(hostname -f)
master_port=7077
master_webui_port=8080
master_url="spark://$master_host:$master_port"
master_webui_url="http://$master_host:$master_webui_port"
# job_id=$(env  | grep SLURM_JOBID)
# master_ip=$(ifconfig | grep -E -o "inet (10\.[0-9\.]+)" | grep -E -o "[0-9\.]+")
spark_dir=$work_dir/spark

# start master node
$spark_dir/sbin/stop-master.sh
rm -f $spark_dir/logs/*
$spark_dir/sbin/start-master.sh --host $master_host --port $master_port --webui-port $master_webui_port

# wait master ready
echo "Waiting master to setup."
sleep 20

# start slaves
touch $work_dir/$flag_filename
if [ ! -d "$work_dir/logs" ]; then
	mkdir $work_dir/logs
fi
rm -f $work_dir/logs/*
for i in $(seq 1 $num_worker) ; do
	echo "Launch worker $i"
	sbatch -N1 -n1 --job-name=${slave_job_name_prefix}$i --time=$allocate_time --cpus-per-task=$allocate_core --mem-per-cpu=$allocate_cpu_mem ./worker.sh $master_url
done

#check worker node allocation
num_log=1
while [ $num_log -ne $(expr $num_worker + 1) ]; do
	sleep 20
	num_log=$(ls -l ${spark_dir}/logs |grep "^-"|wc -l)
	echo "Check worker state: $(expr $num_log - 1) worker(s) are launched"
done

# wait worker nodes ready
echo "Waiting workers to setup."
sleep 20

# check connection
fail_count=0
for log in $spark_dir/logs/*
do
	if [ $(cat $log | grep -E "WARN Worker" | wc -l) -ne 0 ]; then
		echo "$log failed"
		fail_count=$(expr $fail_count + 1)
	fi
done
echo "$fail_count/$num_worker workers are failed to connect to the master."

# finish
squeue -v | grep spark
echo "Setup Spark cluster done."
echo "Cluster summary: $num_worker workers, each worker has ${allocate_core} CPUs, each CPU has ${allocate_cpu_mem} mem."
echo "Spark master URL is $master_url"
echo "Use lynx $master_webui_url to browse cluster info."
