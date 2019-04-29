#!/usr/bin/bash
## auto run in master node

# init
master_url=$1
num_executors=$2
executor_cores=$3
executor_memory=$4
work_dir="/staging/xq/zongjian/"
class_name="ee451s2019.Prepare"
jar_filename="mllib_ver.jar"
wiki_filename="enwiki-20190301-pages-articles-multistream.xml"
num_concepts=10000
num_terms=20000

# compute
cd $work_dir
echo "Environment: $HADOOP_HOME, $SPARK_HOME, $MAHOUT_HOME, $SCALA_HOME"
spark/bin/spark-submit --master $master_url --num-executors $num_executors --executor-cores $executor_cores --executor-memory ${executor_memory}m --class $class_name $jar_filename $wiki_filename $num_concepts $num_terms