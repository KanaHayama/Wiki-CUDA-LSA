#!/usr/bin/bash
## auto run in master node

# init
work_dir="/staging/xq/zongjian/"
class_name="ee451s2019.Prepare"
jar_filename="mllib_ver.jar"
wiki_filename="simplewiki-20190301-pages-articles-multistream.xml"
num_concepts=100
num_terms=20000
output_filename="spark-output.log"
allocate_cpu_mem=$SLURM_MEM_PER_CPU
allocate_core=$SLURM_CPUS_PER_TASK
allocate_node_mem=$(expr $allocate_core \* $allocate_cpu_mem)

master_host=$(hostname -f)
master_port=7077
master_url="spark://$master_host:$master_port"

# compute
echo "Environment: $HADOOP_HOME, $SPARK_HOME, $MAHOUT_HOME, $SCALA_HOME"
$work_dir/spark/bin/spark-submit --master $master_url --executor-memory ${allocate_node_mem}m --class $class_name $jar_filename $wiki_filename $num_concepts $num_terms | tee $output_filename
