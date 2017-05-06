from pyspark.sql.functions import *
import pandas as pd
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.mllib.tree import DecisionTree


print 'Loading data...'
df = sqlContext.read.load('./train.csv', \
                          format='com.databricks.spark.csv', \
                          header=True, \
                          inferSchema=True)

df.cache()

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

# One-hot encode categorical features.
inputCols = [column for column in df.columns if 'cat' in column]
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in df.columns if 'cat' in column] # Takes 30 min.
pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(df).transform(df)
df_r = df_r.select([column for column in df_r.columns if column not in inputCols])

# Vector assembling features.
inputCols_assem = [x for x in df_r.columns if x not in ['id', 'label']]
assembler = VectorAssembler(\
        inputCols = inputCols_assem, \
        outputCol = 'features')
df_r = assembler.transform(df_r)












