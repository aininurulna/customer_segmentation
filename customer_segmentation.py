#import the dataset
import numpy as np
import pandas as pd
from pyspark.sql.functions import unix_timestamp, from_unixtime, to_date
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, dense_rank
from pyspark.sql.functions import unix_timestamp, from_unixtime, to_date
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pylab as pl
data = spark.read.csv("dbfs:/FileStore/tables/joined_cust_sales-3.csv", inferSchema = True, header = True, sep = ",").cache()

#check the data type of each feature
data.printSchema()
data = data.withColumn('new_purchase_date', to_date(unix_timestamp('purchase_date', 'yyyy/MM/dd').cast("timestamp"))).drop('purchase_date')

#sort by cust and purchase date
data = data.orderBy(["cust", "new_purchase_date"], ascending=True)

#create flag of the i-th number of purchase (1st time, 2nd time, etc.) as a new feature
window = Window.partitionBy(data['cust']).orderBy(data['new_purchase_date'])
data = data.select('*', rank().over(window).alias('flag'))

#create a new feature, total spending a customer spends on the client
resultsumtotal = data.groupBy('cust').sum('total_spending').select('cust','sum(total_spending)')

#create a new feature, the number of visit per customer
w = Window.partitionBy('cust')
data = data.withColumn('visit', F.max('flag').over(w)).where(F.col('flag') == F.col('visit')).drop('visit')

data = data.alias('data')
resultsumtotal = resultsumtotal.alias('resultsumtotal')
data2 = resultsumtotal.join(data, resultsumtotal['cust'] == data['cust'], 'left').drop(resultsumtotal['cust'])

#transform the categorical variable
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
stages = []
cat_cols = ["cust", "address", "gender", "status", "ocupation", "flag"]

for cat_col in cat_cols:
  indexer = StringIndexer(inputCol = cat_col, outputCol = cat_col + "_index", stringOrderType = "alphabetAsc")
  encoder = OneHotEncoder(inputCols = [indexer.getOutputCol()], outputCols = [cat_col + "_vec"], dropLast = True)
  stages += [indexer, encoder]

pipeline = Pipeline(stages=stages)
data3 = pipeline.fit(data2).transform(data2)

#build a pipeline of vectorassembler and minmaxscaler
assemblerInputs = ["cust_vec", "address_vec", "gender_vec", "status_vec", "ocupation_vec", "flag_vec", "age", "sum(total_spending)"]

vectorAssembler = VectorAssembler(inputCols = assemblerInputs, outputCol="assembler")

scaler = MinMaxScaler(inputCol="assembler",outputCol="features")


pipeline2 = Pipeline().setStages([
  vectorAssembler,
  scaler,
])

dataready = pipeline2.fit(data3).transform(data3)

# Calculate cost and plot from the elbow method to define the optimum number of clusters
cost = np.zeros(10)
for k in range(2,10):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol('features')
    model = kmeans.fit(dataready)
    cost[k] = model.summary.trainingCost

# Plot the cost
df_cost = pd.DataFrame(cost[2:])
df_cost.columns = ["cost"]
new_col = [2,3,4,5,6,7,8, 9]
df_cost.insert(0, 'cluster', new_col)

import pylab as pl
pl.plot(df_cost.cluster, df_cost.cost)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()

#according to the elbow method, our optimum number of clusters is 3
kmeans = KMeans().setK(3).setSeed(123)
model = kmeans.fit(dataready.select('features'))
predictions = model.transform(dataready)
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))




