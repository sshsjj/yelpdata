from pyspark.ml.feature import Word2Vec
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# vectorSize = 3 , accuracy = 92.3, layers = [3, 5, 4, 3, 2]
# vecotrSize = 5,  accuracy = 94.1, layers = [5, 5, 4, 3, 2]
# vectorSize = 10, accuracy = 94.8, layers = [10, 5, 4, 3, 2]
# vectorSize = 15, accuracy = 95.0, layers = [15, 5, 4, 3, 2]
# vecotrSize = 20, accuracy = 95.3, layers = [20, 5, 4, 3, 2]
# vecotrSize = 30, accuracy = 95.3, layers = [30, 15, 10, 5, 4, 3, 2]
# vecotrSize = 30, accuracy = 95.47, layers = [30, 6, 4, 2]  {layers = [30, 10, 4, 2] lower 95.45}
# vecotrSize = 30, accuracy = 95.36, layers = [30, 5, 2]   lower 
# vecotrSize = 50, accuracy = 95.91, layers = [50, 6, 4, 2] reduced 1 layer and get higer score
# vecotrSize = 100, accuracy = 96.07, layers = [100, 6, 4, 2], 
# vecotrSize = 200, accuracy = 96.43, layers = [200, 6, 4, 2], 
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="words1")
word2vec = Word2Vec(vectorSize=200, minCount=0, inputCol="words1", outputCol="features")
layers = [200, 6, 4, 2] 
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=2000)
nn_pipeline = Pipeline(stages=[tokenizer,remover, word2vec, trainer])

pipelineFit = nn_pipeline.fit(train_set)
predictions = pipelineFit.transform(val_set)

predictionAndLabels = predictions.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
