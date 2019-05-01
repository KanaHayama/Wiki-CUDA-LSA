## manually run in login node

# init
work_dir="/staging/xq/zongjian/"
allocate_time="23:59:59"
allocate_core=8
allocate_cpu_mem="4g"
master_job_name="spark-master"

# python environment
source /usr/usc/python/default/setup.sh

# allocate master node
cd $work_dir
rm -f $output_filename
salloc -N1 -n1 --job-name=$master_job_name --time=$allocate_time --cpus-per-task=$allocate_core --mem-per-cpu=$allocate_cpu_mem 

echo "Welcome back"