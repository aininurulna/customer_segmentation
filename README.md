# customer_segmentation
Customer Segmentation using Pyspark

A Kmeans clustering implemented in PySpark. The pyspark.ml.clustering library was used to import the Kmeans. As an initial step before creating the model, the elbow method was conducted to find the optimum number of clusters. The result obtained is 3, thus the argument setK = 3 was passed to the object.

Before the modelling, a pipeline was created to do one-hot encoding on the features. Since there are some string type features, we begin by using StringIndexer to first index the features and we then pass the results to OneHotEncoder which runs one-hot encoding on the features and stores the results as a sparse matrix, encoded using vectortype.

After that, another pipeline which contains VectorAssembler and MinMaxScaler is created. VectorAssembler: Assemble the feature columns into a feature vector. MinMaxScaler: rescaling each feature to a specific range ([0,1]).

The data is a dummy joined POS retail data and customer profile of 200 points generated by myself. The columns:
cust: customer id\
address: ward of the address\
gender: M or F\
status: married or single\
ocupation: job\
age: age\
product: product purchased by the customer\
quantity: number of product purchased at one transaction\
total_spending: total spending at one transaction\
purchase_date: date of purchase
