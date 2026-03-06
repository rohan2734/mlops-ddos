from pyspark.sql import SparkSession,Row

spark = SparkSession.builder.appName("sparkDF").getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# build a session with specified configuration
# set a human readable name for spark session
# retrieve an existing session if available

print(spark)

my_rowA = Row(name="Peter",age=22,location="Florida")
# define a row object by specified named arguments , corresponding to the column names in the dataframe
print("Name:",my_rowA.name)
print("Age:",my_rowA['age'])
print("Location:",my_rowA[2])

my_rowB = Row(name="Mark",age=25,location=None)
# we must note that declaring row objects of some standard schema, we are not allowed to omit a named argument, this is not allowed
# they must be explicitly none in such instances

my_rowC = Row(name="Joey",age=26,location="London")

Student = Row("name","age","location")
my_row= Student("Peter",22,"Florida")

print("name:",my_row.name)
print("age:",my_row.age)
print("location:",my_row.location)


# create pyspark dataframes
## from list of lists

data = [("Peter",22,"Florida"),
        ("Joey",26,"London"),
        ("Mark",25,"Dubai"),
        ("Kate",23,"Sydney")]

spark_df = spark.createDataFrame(data)

print(type(spark_df))
spark_df.show()

spark_df2 = spark.createDataFrame(data).toDF("Name","Age","Location")
spark_df2.printSchema()

from pyspark.sql.types import StructType , StructField, StringType, IntegerType

schema = StructType([
    StructField("Name",StringType(),True),
    StructField("Age",IntegerType(),True),
    StructField("Location",StringType(),True)])

data = [("Peter",22,"Florida"),
        ("Joey",26,"London"),
        ("Mark",25,"Dubai"),
        ("Kate",23,"Sydney")]

spark_df = spark.createDataFrame(data,schema=schema)

from pyspark.sql import Row

data = [Row(name="Peter",age=22,location="Florida"),
        Row(name="Joey",age=26,location="London"),
        Row(name="Mark",age=25,location="Sydney")]

spark_df = spark.createDataFrame(data,schema=schema)

spark_df.show()

# spark_df = spark.read.table("employee_dataset_csv")
#
# spark_df
#
# spark_df = spark.read.format("csv").load("data.csv")

spark_df.printSchema()

spark_df = spark.read.format("csv").option("inferSchema","true").option("header","true").load("data.csv")

spark_df = spark.read.format("parquet").option("inferSchema","true").option("header","true").load("data.parquet")

#in case of parquet format, we dont need to mention the options, format , we can just run read.load or read.parquet
spark_df = spark.read.load("data.parquet")
spark_df.printSchema()

spark_df = spark.read.parquet("data.parquet")
spark_df.printSchema()
# parquet is a columnar storage file format that inherently maintains schema information within the data files
# unlike the CSV or other formats, where the schema may need to be inferred or explicitly provided during read operation, parquet files store schema details alongside the actual data

# instead of iterating through files using a for loop, and then concatenating , we can also read a directory of files
df = spark.read.load("path/to/directory",format="parquet",pathGlobFilter="*.parquet")

empty_df = spark.createDataFrame([],"col1 INt, col2 STRING")

empty_df.show()

data = [("Peter",22,"Florida"),
        ("Joey",26,"London"),
        ("Mark",25,"Dubai"),
        ("Kate",23,"Sydney")]

spark_df = spark.createDataFrame(data,schema="Name STRING,Age INT,Location STRING")

spark_df.show(n=20,truncate=True,vertical=False)
# number of rows to display, truncate long text columns, vertical display orientation
spark_df.show(2) # first two rows of spark dataframe

#when truncate=True , this argument truncates long strings in the displayed dataframe to fit within the column width ,if set to False, it displays complete content of each cell,which can be useful for inspecting long text fields
# vertical(default=False) , when set to True, the vertical argument displays the DataFrame in vertical orientation, which is helpful when dealing with wide DataFrames

# show method just prints, but it doesnt return anything
# head and tail are indended for that purpose
# first k rows -? head(k)
spark_df.head(2) # returns k rows of dataframe as list of Row Objects

# get output of first  k rows as pyspark dataframe
spark_df.limit(2).show()
spark_df.tail(2)

new_df = spark.createDataFrame(spark_df.tail(2),schema=spark_df.schema)
new_df.show()

row_count = spark_df.count()

spark_df.select(spark_df["Salary"]*1.1).show()