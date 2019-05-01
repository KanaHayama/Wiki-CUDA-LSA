#!/usr/bin/bash
## auto run in master node

# init
cores_per_worker=$1
if [ $cores_per_worker == "" ]; then
	echo "Worker CPU not set!"
	return -1
fi

mem_per_worker=$2
if [ $mem_per_worker == "" ]; then
	echo "Worker MEM not set!"
	return -1
fi

wiki_filename="simplewiki-20190301-pages-articles-multistream.xml"
wiki_filename="enwiki-20190301-pages-articles-multistream.xml"
num_concepts=100
num_terms=20000

work_dir="/staging/xq/zongjian/"

jar_filename="mllib_ver.jar"
output_filename="spark-output.log"

master_host=$(hostname -f)
master_port=7077
master_url="spark://$master_host:$master_port"

# summary
echo "Summary:"
echo "Use $cores_per_worker cores per worker, $mem_per_worker mem per worker"
echo "Input file is $wiki_filename"
echo "Reduce to $num_concepts concepts, and $num_terms terms"
echo "Output record to $output_filename"

# compute
class_name="ee451s2019.CPU"
$work_dir/spark/bin/spark-submit --master $master_url --executor-cores $cores_per_worker --executor-memory ${mem_per_worker} --class $class_name $jar_filename $wiki_filename $num_concepts $num_terms | tee $output_filename
class_name="ee451s2019.GPU"
$work_dir/spark/bin/spark-submit --master $master_url --executor-cores $cores_per_worker --executor-memory ${mem_per_worker} --class $class_name $jar_filename $wiki_filename $num_concepts $num_terms | tee $output_filename