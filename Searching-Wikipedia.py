from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# creating a local spark configuration
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf) # creates the rdd

# Load documents (one per line).
rawData = sc.textFile("/Users/zoserreaha/Desktop/MLCourse/subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t")) # splits each document into list
documents = fields.map(lambda x: x[3].split(" ")) # splits body of text into list of words

# Store the document names for later:
documentNames = fields.map(lambda x: x[1]) # field 1 is the name of the document

# hashing the words in each document to 1 of 100,000 numerical values
hashingTF = HashingTF(100000)  #100K hash buckets just to save some memory
tf = hashingTF.transform(documents) # converts list of words into hash values that represent each word


# Computes the TF*IDF of each term in each document:
tf.cache()
idf = IDF(minDocFreq=2).fit(tf) # ignores words that don't appear at least twice
tfidf = idf.transform(tf) # computes the TF*IDF score for each word in the document


# An article for "Abraham Lincoln" is in the data
# set, so let's search for "Gettysburg" (Lincoln gave a famous speech there):


gettysburgTF = hashingTF.transform(["Gettysburg"]) # transforms into hash value
gettysburgHashValue = int(gettysburgTF.indices[0])

# Extracts the TF*IDF score for Gettsyburg's hash value into
# a new RDD for each document:
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])

# Zips the document names so we can see which is which:
zippedResults = gettysburgRelevance.zip(documentNames)

# Prints the document with the maximum TF*IDF value:
print("Best document for Gettysburg is:")
print(zippedResults.max())
