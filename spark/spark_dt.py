from pyspark.sql.functions import *
import pandas as pd
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree

print 'Loading data...'
df = sqlContext.read.load('./train.csv', \
                          format='com.databricks.spark.csv', \
                          header=True, \
                          inferSchema=True)

print pd.DataFrame(df.take(5), columns=df.columns).transpose()
print df.describe().toPandas().transpose()

df = df.withColumnRenamed('loss', 'label')

df.select(log1p(df.label).alias('log_label')).rdd.map(lambda l: l.log_label).collect() # This just prints the log1p transformed column
df = df.withColumn('label', log1p('label')) # This assigns log1p transformed data back to df
df = df.drop('id')

# This plot the numerical features. But we don't plot it because we are using server.
# numerical_features = [t[0] for t in df.dtypes if t[1] == 'int' or t[1] == 'double']
# data_to_plot = df.select(numerical_features).toPandas()
# axs = pd.scatter_matrix(data_to_plot, figsize=(12, 12))

# TODO: Need to one-hot encode categorical features.

# Label the data
def labelData(data):
    return data.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))

training_data, testing_data = labelData(df).randomSplit([0.8, 0.2])
model = DecisionTree.trainClassifier(training_data, numClasses=2, maxDepth=2,
                                     categoricalFeaturesInfo={1:2, 2:2},
                                     impurity='gini', maxBins=32)












