# Yelp dataset - Dentist Review Practice 
# Running on Databricks Cluster(Conmmunity Version)
file_location = "/FileStore/tables/dentist_reviews_clean.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = "`"

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# Create TempView of df
temp_table_name = "dentist_reviews_clean_csv"
df.createOrReplaceTempView(temp_table_name)

# Labele the reviews which stores in stars_y column (>=4.0 : 1, <4.0 : 0), filter out the non-word characters and merge multiple spaces into one
light_df = spark.sql("""select case when stars_y >= 4.0 then 1 else 0 end as label, regexp_replace(regexp_replace(lower(text),'(\\\W)', ' '), ' +', ' ') as text from dentist_reviews_clean_csv""")
display(light_df)

# Spark ML Pipeline
from pyspark.ml.feature import StringIndexer
# Since I have already labeled the data from the spark sql, String Indexer was not used
# Example: label_str = StringIndexer(inputCol = "result", outputCol = "label")
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline


(train_set, val_set) = light_df.randomSplit([0.7, 0.3], seed = 2000)

tokenizer = Tokenizer(inputCol="text", outputCol="words")
# hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2**16)
cv = CountVectorizer(vocabSize=2**16, inputCol="words", outputCol='cv')
idf = IDF(inputCol="cv", outputCol="features", minDocFreq = 5)
lr = LogisticRegression(maxIter=100)
#pipeline = Pipeline(stages=[tokenizer, hashtf, idf])
pipeline = Pipeline(stages=[tokenizer, cv, idf, lr])
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

pipelineFit = pipeline.fit(train_set)
predictions = pipelineFit.transform(val_set)
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
roc_auc = evaluator.evaluate(predictions)




