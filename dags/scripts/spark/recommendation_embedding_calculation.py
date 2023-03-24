from elasticsearch import Elasticsearch
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, array_contains, lit, current_timestamp, unix_timestamp
from pyspark.ml.evaluation import RegressionEvaluator


# test your ES instance is running

def get_es(es_address, es_user, es_password):
    es = Elasticsearch(hosts=[es_address], http_auth=(es_user, es_password), verify_certs=False)
    return es
    
def create_index(es, vector_dim, index_name):
    mapping = {
        # this mapping definition sets up the metadata fields for the products
        "mappings": {
            "properties": {
                "item_id": {
                    "type": "keyword"
                },
                "category": {
                    "type": "keyword"
                },
                "description": {
                    "type": "keyword"
                },
                "unit_price": {
                    "type": "keyword"
                },
                # the following fields define our model factor vectors and metadata
                "model_factor": {
                    "type": "dense_vector",
                    "dims" : vector_dim
                },
                "model_version": {
                    "type": "keyword"
                },
                "model_timestamp": {
                    "type": "date"
                }          
            }
        }
    }
    if es.indices.exists(index=index_name):
        response = es.indices.create(index=index_name, body=mapping)
    else:
        response = 'indices already existed'
    return response

def write_to_index(dataframe, index_name, _id='item_id', write_operation='index', mode='append'):
    dataframe.write.format('org.elasticsearch.spark.sql') \
    .options(**{
        'es.mapping.id': _id,
        'es.nodes.wan.only': 'true',
        'es.write.operation': write_operation
    }).mode(mode).save(index_name)

def recommend_vector(
    input_loc: str, output_loc: str, run_id: str, 
    es_address: str, es_user: str, es_password: str, 
    vector_dim: int, index_name:str
) -> None:
    es = get_es(es_address, es_user, es_password)
    response = create_index(es, vector_dim, index_name)
    spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

    df_raw = spark.read.option("header", True).csv(input_loc)
    df_cleaned = df_raw.select(
        col('StockCode').substr(0, 5).cast("int").alias('item_id'), 
        col('Description').alias('description'), 
        col('CustomerID').cast("int").alias('customer_id'),
        col('UnitPrice').alias('unit_price')
    ).filter(col('item_id').isNotNull()) \
    .filter(col('customer_id').isNotNull())
    
    write_to_index(
        df_cleaned.select(col('item_id'), col('category'), col('description'), col('unit_price')), 
        index_name)
    
    
    df_ratings = df_cleaned.select(
        col('customer_id'),
        col('item_id'),
        lit(1).alias('rating')
    )
    train, test = df_ratings.randomSplit([0.8, 0.2])
    
    als = ALS(
        maxIter = 5,
        regParam = 0.01,
        userCol = "customer_id",
        itemCol = "item_id",
        ratingCol = "rating",
        coldStartStrategy = "drop",
        seed = 0
    )

    model = als.fit(train)
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("RMSE="+str(rmse))
    
    
    model = als.fit(df_ratings)
    ver = model.uid
    ts = unix_timestamp(current_timestamp())
    embedding_vectors = model.itemFactors.select(
        col("id").alias('item_id'),
        col("features").alias("model_factor"),
        lit(ver).alias("model_version"),
        ts.alias("model_timestamp")
    )
    
    write_to_index(
        embedding_vectors, 
        index_name,
        index_name, 
        write_operation='upsert', 
    )