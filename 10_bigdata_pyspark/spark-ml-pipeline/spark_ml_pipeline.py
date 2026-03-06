from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# -----------------------
# Spark session
# -----------------------
spark = (
    SparkSession.builder
    .appName("SimpleSparkMLPipeline")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# -----------------------
# Data generation
# -----------------------
n_rows = 10_000

start_ts = F.unix_timestamp(F.lit("2024-01-01 00:00:00"))  # in seconds
df = (
    spark.range(n_rows)
    .withColumn("ts", start_ts + (col("id") * 60))  # 1-minute intervals
    .withColumn("ds", F.from_unixtime(col("ts")).cast("timestamp"))
    .withColumn("feature_a", F.randn(seed=42))
    .withColumn("feature_b", F.rand(seed=1337) * 10.0)
    .withColumn("y", 2.0 * col("feature_a") + 0.3 * col("feature_b") + F.randn(seed=7) * 0.5)
    .drop("ts")
)

# -----------------------
# Time-based split
# -----------------------
w = Window.orderBy("ds")
df_ranked = df.orderBy("ds").withColumn("rank", F.row_number().over(w))

n = df_ranked.count()
cutoff = int(n * 0.8)

train = df_ranked.filter(F.col("rank") <= cutoff).drop("rank")
test  = df_ranked.filter(F.col("rank") >  cutoff).drop("rank")

# -----------------------
# Pipeline
# -----------------------
# Create features AFTER the split (avoids leakage)
train = train.withColumn("hour", F.hour("ds").cast("double"))
test = test.withColumn("hour", F.hour("ds").cast("double"))

imputer = Imputer(
    inputCols=["hour", "feature_a", "feature_b"],
    outputCols=["hour_imp", "feature_a_imp", "feature_b_imp"]
)

assembler = VectorAssembler(
    inputCols=["hour_imp", "feature_a_imp", "feature_b_imp"],
    outputCol="features"
)

lr = LinearRegression(
    labelCol="y",
    featuresCol="features"
)

pipeline = Pipeline(stages=[imputer, assembler, lr])
model = pipeline.fit(train)

# -----------------------
# Evaluation
# -----------------------
preds_test = model.transform(test)
evaluator = RegressionEvaluator(labelCol="y", predictionCol="prediction", metricName="r2")
r2_test = evaluator.evaluate(preds_test)
print(f"R²(test): {r2_test:.4f}")

spark.stop()