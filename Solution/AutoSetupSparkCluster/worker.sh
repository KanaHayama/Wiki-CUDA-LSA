#!/usr/bin/bash
#SBATCH --output="/staging/xq/zongjian/logs/%j.out"
#SBATCH --error="/staging/xq/zongjian/logs/%j.err"

## auto run in slave node

#init
work_dir="/staging/xq/zongjian/"
master_url=$1
flag_filename="spark_cluster_running"
#self_ip=$(ifconfig | grep -E -o "inet (10\.[0-9\.]+)" | grep -E -o "[0-9\.]+")
spark_dir=$work_dir/spark

# launch worker
echo "Connect to $master_url"
$spark_dir/sbin/stop-slave.sh
$spark_dir/sbin/start-slave.sh $master_url

# finish
echo "Worker at $SLURMD_NODENAME launched"

# loop to keep alive
while [ -f "$work_dir/$flag_filename" ]; do
	sleep 30
	echo "Beap"
done