## Script to install softwares into /staging dir, since files in this dir may be deleted by admin

# init
work_dir="/staging/xq/zongjian/"

# install Hadoop
cd $work_dir
if [ ! -d "hadoop" ];then
	if [ ! -f "hadoop-2.9.2.tar.gz" ]; then
		wget https://www-us.apache.org/dist/hadoop/common/hadoop-2.9.2/hadoop-2.9.2.tar.gz
	fi
	tar -xzf hadoop-2.9.2.tar.gz
	mv hadoop-2.9.2 hadoop
else
	echo "Hadoop exists"
fi
HADOOP_HOME=$work_dir/hadoop/
export HADOOP_HOME
echo "Hadoop home set to $HADOOP_HOME"
PATH=$PATH:$HADOOP_HOME/bin
export PATH

# install Spark (spark 2.4 and later is built using scala 2.12, Mahout now do not support scala 2.12)
cd $work_dir
if [ ! -d "spark" ];then
	if [ ! -f "spark-2.3.3-bin-hadoop2.7.tgz" ]; then
		wget https://www-eu.apache.org/dist/spark/spark-2.3.3/spark-2.3.3-bin-hadoop2.7.tgz
	fi
	tar -xzf spark-2.3.3-bin-hadoop2.7.tgz
	mv spark-2.3.3-bin-hadoop2.7 spark
else
	echo "Spark exists"
fi
if [ ! -f "spark/conf/log4j.properties" ]; then
	cp log4j.properties spark/conf/log4j.properties
fi
SPARK_HOME=$work_dir/spark/
export SPARK_HOME
echo "Spark home set to $SPARK_HOME"
PATH=$PATH:$SPARK_HOME/bin
export PATH

# install Scala (Mahout not support latest version, now only 2.11)
cd $work_dir
if [ ! -d "scala" ];then
	if [ ! -f "scala-2.11.12.tgz" ]; then
		wget https://downloads.lightbend.com/scala/2.11.12/scala-2.11.12.tgz
	fi
	tar -xzf scala-2.11.12.tgz
	mv scala-2.11.12 scala
else
	echo "Spark exists"
fi
SCALA_HOME=$work_dir/scala/
export SCALA_HOME
echo "Scala home set to $SCALA_HOME"
PATH=$PATH:$SCALA_HOME/bin
export PATH

# install maven
cd $work_dir
if [ ! -d "maven" ];then
	if [ ! -f "apache-maven-3.6.1-bin.tar.gz" ]; then
		wget https://www-eu.apache.org/dist/maven/maven-3/3.6.1/binaries/apache-maven-3.6.1-bin.tar.gz
	fi
	tar -xzf apache-maven-3.6.1-bin.tar.gz
	mv apache-maven-3.6.1 maven
else
	echo "Maven exists"
fi
MAVEN_PATH=$work_dir/maven/
export MAVEN_PATH
echo "Scala home set to $MAVEN_PATH"
PATH=$PATH:$MAVEN_PATH/bin
export PATH
cd $work_dir
if [ ! -d ".m2" ];then
	mkdir .m2
fi

# install Mahout
cd $work_dir
if [ ! -d "mahout" ]; then
	if [ ! -f "mahout-0.14.0-source-release.zip" ]; then 
		wget http://www.apache.org/dist/mahout/0.14.0/mahout-0.14.0-source-release.zip
	fi
	unzip mahout-0.14.0-source-release.zip
	mv mahout-0.14.0 mahout
else
	echo "Mahout exists"
fi
MAHOUT_HOME=$work_dir/mahout/
export MAHOUT_HOME
echo "Mahout home set to $MAHOUT_HOME"
PATH=$PATH:$MAHOUT_HOME/bin
export PATH

# download data
cd $work_dir
if [ ! -f "simplewiki-20190301-pages-articles-multistream.xml" ]; then
	wget https://dumps.wikimedia.org/simplewiki/20190301/simplewiki-20190301-pages-articles-multistream.xml.bz2
	bunzip2 simplewiki-20190301-pages-articles-multistream.xml.bz2
fi
if [ ! -f "enwiki-20190301-pages-articles-multistream.xml" ]; then
	wget https://dumps.wikimedia.org/enwiki/20190301/enwiki-20190301-pages-articles-multistream.xml.bz2
	bunzip2 enwiki-20190301-pages-articles-multistream.xml.bz2
fi

# download & build ViennaCL
cd $work_dir
if [ ! -d "viennacl-dev" ]; then
	git clone https://github.com/viennacl/viennacl-dev.git
	cd viennacl-dev
	mkdir build
	cd build
	# ENABEL_CUDA=yes, ENABEL_OPENCL=no, -arch=sm_30
	# ccmake .. 
	cmake ..
	make
	# head files in ../viennacl
	# Add libviennacl.so to LD_LIBRARY_PATH
else
	echo "Viennacl exists"
fi

# build CUDA Mahout
# copy core_ & spark_ to CLASSPATH

# download repo
cd $work_dir
if [ ! -d "Wiki-CUDA-LSA" ]; then
	git clone https://github.com/KanaHayama/Wiki-CUDA-LSA.git
fi
cd Wiki-CUDA-LSA
git pull https://github.com/KanaHayama/Wiki-CUDA-LSA.git

# compile jars
rm -f $work_dir/mllib_ver.jar
cd $work_dir/Wiki-CUDA-LSA/Solution/Wiki-LSA
mvn clean package -P special
ln -s $work_dir/Wiki-CUDA-LSA/Solution/Wiki-LSA/target/Wiki-LSA-1.0.0-jar-with-dependencies.jar $work_dir/mllib_ver.jar

