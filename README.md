# E-commerce Data Batch Processing pipeline
* extended from https://github.com/josephmachado/beginner_de_project


- [E-commerce Data Batch Processing pipeline](#e-commerce-data-batch-processing-pipeline)
  - [Design](#design)
  - [Extra Features](#extra-features)
  - [ERROR HANDLING](#error-handling)

## Design
![Data pipeline design](assets/images/new_design.png)

We will be using Airflow to orchestrate

1. Classifying movie reviews with Apache Spark.
2. Loading the classified movie reviews into the data warehouse.
3. Extract user purchase data from an OLTP database and load it into the data warehouse.
4. Joining the classified movie review data and user purchase data to get `user behavior metric` data.
![asd](assets/images/Screenshot2023-03-22.png)
* NOTE: start the same airflow docker locally, then use port forwarding to send request to airflow docker running on EC2

## Extra Features
### 1. Adding Extra Airflow Steps
a. Install ElasticSearch on EMR 
  * #TODO: modify terraform EMR
  
b. Calculate recommendataion embedding of products using AWS EMR and save to EMR ES 
  * #TODO: modify spark submit jar file
  * #TODO: is there a way that EMR can save results to OpenSearch directly

c. recommendation api query EMR ES
d. (optional) use singlestore https://medium.com/@VeryFatBoy/using-singlestore-spark-and-alternating-least-squares-als-to-build-a-movie-recommender-system-6e74f4e5908d



### 2. Recommendataion api serving 
a. Build a real-time recommendation api using fastapi and elasticsearch or singlestore 
  * #TODO: dockerize and how did container request elasticsearch on EMR?

### 3. Spark data skew analysis and optimization
* #TODO

## ERROR HANDLING
1. Error1: make infra-up
   
         could not start transaction: dial tcp xx.xxx.xxx.xx:5439: connect: operation timed out │ │ with redshift_schema.external_from_glue_data_catalog, │ on main.tf line 170, in resource "redshift_schema" "external_from_glue_data_catalog": │ 170: resource "redshift_schema" "external_from_glue_data_catalog"
   * Solved it by editing the Source of the Inbound Rules of the EC2 Default Security Group (SG) to my IP Address. Just get into EC2 Dashboard via AWS Console and the list of SG is under the Network & Security tab. There is multiple SG and you have to edit the 'default' EC2 SG only.
    https://aws.amazon.com/premiumsupport/knowledge-center/private-redshift-cluster-local-machine/
    https://stackoverflow.com/questions/67729058/add-a-security-group-to-the-inbound-rule-of-another-security-group-as-a-source-w

2. Error2: make cloud-airflow 
   
         bind [127.0.0.1]:8082: Address already in use
         channel_setup_fwd_listener_tcpip: cannot listen to port: 8082
         Could not request local forwarding.
   * kill process (lsof -i:port) listening on port first

3. Error3: Airflow Error
   
         could not connect to server: Connection timed out 
         Is the server running on host "alande-redshift-cluster.c2uknbxe8vtj.us-east-1.redshift.amazonaws.com" (172.31.0.228) and accepting TCP/IP connections on port 5439
   * 檢查 redshift security group inbound rule 是不是長這樣 
     ![](assets/images/QXRq2.png)

   * how to make terraform aws_security_group type = Redshift???
4. Error4: elasticsearch spark error

          An error occurred while calling o434.save.: org.elasticsearch.hadoop.EsHadoopIllegalArgumentException: Cannot detect ES version - typically this happens if the network/Elasticsearch cluster is not accessible or when targeting a WAN/Cloud instance without the proper setting 'es.nodes.wan.only'
          // log in docker
          received plaintext http traffic on an https channel, closing connection Netty4HttpChannel{localAddress=/172.21.0.2:9200, remoteAddress=/172.21.0.1:5600  
   * disable security
  
          docker run --name es01 --net elastic --memory=8g -p 9200:9200 -e "xpack.security.enabled=false" -e "node.name=node-1" -e "cluster.initial_master_nodes=node-1" -it docker.elastic.co/elasticsearch/elasticsearch:8.6.1 