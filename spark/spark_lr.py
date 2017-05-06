from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
import pandas as pd
import numpy as np


print 'Starting Data Preprocessing...'
# sqlContext = SQLContext(sc)
# train = pd.read_csv('./train.csv') # This will read csv file from local, not remote on hadoop.
# data = sqlContext.createDataFrame(train) # This will take some time.
# print data.printSchema()

# Different way to load CSV, using the Spark CSV package. Note this reads the remote HDFS storage not local like above.
df = sqlContext.read.load('./train.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')
df = df.cache()
# Rename 'loss' column as 'label'.
df = df.withColumnRenamed('loss', 'label')
print df.printSchema()

# Need to take Log1p of 'label' column.
df = df.withColumn('label', log1p('label'))
df = df.drop('id')

# One hot encoding categorical features.
inputCols = [column for column in df.columns if 'cat' in column]
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in df.columns if 'cat' in column] # Takes 30 min.
pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(df).transform(df)
print df_r.printSchema()

# Remove old categorical features that are not one-hot encoded.
df_r = df_r.select([column for column in df_r.columns if column not in inputCols])

# Vector assembling features.
inputCols_assem = [x for x in df_r.columns if x not in ['id', 'label']]
assembler = VectorAssembler(\
        inputCols = inputCols_assem, \
        outputCol = 'features')
df_r = assembler.transform(df_r)

# Train test split the Data.
train, test = df_r.randomSplit([0.8, 0.2], seed=12345)
print 'Finished data preprocessing...'

# Fitting & predicting pipeline.
evaluator = RegressionEvaluator(metricName="mae")
# lr = LinearRegression().setSolver("l-bfgs")
lr = LinearRegression()
grid = ParamGridBuilder().addGrid(lr.maxIter, [500]) \
                         .addGrid(lr.regParam, [0]) \
                         .addGrid(lr.elasticNetParam, [1]) \
                         .build()
lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, \
                       evaluator=evaluator, numFolds=3)
lrModel = lr_cv.fit(train) # Takes 30 min to run.
bestModel = lrModel.bestModel
print 'MAE: ', lrModel.avgMetrics
print 'Best Param (regParam): ', bestModel._java_obj.getRegParam()
print 'Best Param (MaxIter): ', bestModel._java_obj.getMaxIter()
print 'Best Param (elasticNetParam): ', bestModel._java_obj.getElasticNetParam()
print 'Param Map: ', bestModel._java_obj.extractParamMap()
bestModel.save('./BestLinearModel') # Note: this will save on hadoop not local.

# Predict on the hold out test set, and check the accuracy.
pred_transformed_data = bestModel.transform(test)
pred_score = evaluator.evaluate(pred_transformed_data)
print evaluator.getMetricName(), 'accuracy: ', pred_score







