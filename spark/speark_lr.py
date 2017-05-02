from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
import pandas as pd


print 'Starting Data Preprocessing...'
sqlContext = SQLContext(sc)
train = pd.read_csv('./train.csv') # This will read csv file from local, not remote on hadoop.
data = sqlContext.createDataFrame(train)
print data.printSchema()

# Rename 'loss' column as 'label'
df = data.select([col(s).alias('label') if s == 'loss' else s for s in data.columns])
print df.printSchema()

# One hot encoding categorical features.
inputCols = [column for column in df.columns if 'cat' in column]
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in df.columns if 'cat' in column]
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
lr = LinearRegression(regParam=0.9, elasticNetParam=0.0).setSolver("l-bfgs")
grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 10]).build()
lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, \
                       evaluator=evaluator, numFolds=3)
lrModel = lr_cv.fit(train)
bestModel = lrModel.bestModel
print 'MAE: ', lrModel.avgMetrics
bestModel.save('./BestLinearModel') # Note: this will save on hadoop not local.

# Predict on the hold out test set, and check the accuracy.
pred_transformed_data = lrModel.transform(test)
pred_score = evaluator.evaluate(pred_transformed_data)
print evaluator.getMetricName(), 'accuracy: ', pred_score
