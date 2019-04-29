#!/usr/bin/bash
#SBATCH --output="/staging/xq/zongjian/logs/%j.out"
#SBATCH --error="/staging/xq/zongjian/logs/%j.err"

## auto run in slave node

#init
work_dir="/staging/xq/zongjian/"
master_url=$1
#self_ip=$(ifconfig | grep -E -o "inet (10\.[0-9\.]+)" | grep -E -o "[0-9\.]+")

# launch slave
echo "Connect to $master_url"
cd $work_dir/spark/sbin
./stop-slave.sh
./start-slave.sh $master_url

# finish
echo "Worker at $SLURMD_NODENAME started"

# loop to keep alive
while true; do
	sleep 60
	echo "Beap"
done