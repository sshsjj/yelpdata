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

# Labeled the reviews which stores in stars_y column (>=4.0 : 1, <4.0 : 0), filtered out the non-word characters and merge multiple spaces into one
light_df = spark.sql("""select case when stars_y >= 4.0 then 1 else 0 end as label, regexp_replace(regexp_replace(lower(text),'(\\\W)', ' '), ' +', ' ') as text from dentist_reviews_clean_csv""")
display(light_df)

# Spark ML Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression

(train_set, val_set) = light_df.randomSplit([0.7, 0.3], seed = 2000)

tokenizer = Tokenizer(inputCol="text", outputCol="words")
cv = CountVectorizer(vocabSize=2**16, inputCol="words", outputCol='cv')
idf = IDF(inputCol="cv", outputCol="features", minDocFreq = 5)
lr = LogisticRegression(maxIter=100)
pipeline = Pipeline(stages=[tokenizer, cv, idf, lr])

pipelineFit = pipeline.fit(train_set)
predictions = pipelineFit.transform(val_set)
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
roc_auc = evaluator.evaluate(predictions)




