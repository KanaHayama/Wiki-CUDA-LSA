param (
	[bool]$local_mode = $true,
	[bool]$debug_mode = $true,
	[string]$jar_filename = "target/Wiki-LSA-1.0.0-jar-with-dependencies.jar",
	[string]$class_name = "ee451s2019.Prepare",
	[string]$wiki_filename = "D:/TEST/simplewiki-20190301-pages-articles-multistream.xml",
	[int]$debug_port = 5005,
	[string]$debug_wait_debugger = "y"
)

# MAHOUT -> Spark
$env:MAHOUT_LOCAL="true"
$env:MASTER="local[*]"
# IntelliJ remote debugger
if ($debug_mode) {
	$env:SPARK_SUBMIT_OPTS = "-agentlib:jdwp=transport=dt_socket,server=y,suspend={0},address={1}" -f $debug_wait_debugger, $debug_port
}
# submit
if ($local_mode) {
	spark-submit --driver-memory 16g --class $class_name $jar_filename $wiki_filename
} else {
	spark-submit --executor-memory 64g --num-executors 1 --executor-cores 16 --class $class_name $jar_filename $wiki_filename
}
return $true