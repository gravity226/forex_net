import pyspark as ps
import json

# Uses all 4 cores on your machine
sc = ps.SparkContext('local[4]')

# sc.textFile('jsons/AUDJPY_20110606_01-00-00.json').cache()
# sc.textFile('jsons/AUDJPY_20130822_12-00-00.json').cache()

files = [   "AUDJPY_20110606_01-00-00.json",	"GBPJPY_20100819_19-00-00.json",
            "AUDJPY_20130822_12-00-00.json",	"GBPJPY_20140923_11-00-00.json",
            "AUDUSD_20081229_03-00-00.json",	"GBPUSD_20040810_11-00-00.json",
            'CHFJPY_20030501_23-00-00.json',	"GBPUSD_20080229_20-00-00.json",
            "CHFJPY_20050713_03-00-00.json",	'GBPUSD_20131008_23-00-00.json',
            "CHFJPY_20050916_19-00-00.json",	"NZDUSD_20030314_06-00-00.json",
            "CHFJPY_20071005_03-00-00.json",	"NZDUSD_20060425_02-00-00.json",
            "CHFJPY_20090513_23-00-00.json",	"USDCAD_20010703_05-00-00.json",
            "EURCHF_20011023_01-00-00.json",	"USDCAD_20080902_03-00-00.json",
            "EURCHF_20070809_09-00-00.json",	"USDCAD_20081224_20-00-00.json",
            "EURCHF_20081210_19-00-00.json",	"USDCAD_20090416_15-00-00.json",
            "EURCHF_20140613_02-00-00.json",	'USDCAD_20110531_22-00-00.json',
            'EURGBP_20061205_17-00-00.json',	"USDCHF_20020531_10-00-00.json",
            "EURGBP_20130411_16-00-00.json",	"USDCHF_20021211_12-00-00.json",
            "EURJPY_20020902_11-00-00.json",	'USDCHF_20031127_01-00-00.json',
            "EURJPY_20130107_03-00-00.json",	"USDCHF_20050819_18-00-00.json",
            "EURJPY_20130404_08-00-00.json",	"USDCHF_20110221_09-00-00.json",
            "EURJPY_20140124_10-00-00.json",	"USDCHF_20110317_05-00-00.json",
            "EURUSD_20041231_15-00-00.json",	'USDCHF_20110519_04-00-00.json',
            "EURUSD_20061026_19-00-00.json",	'USDJPY_20120314_08-00-00.json',
            "EURUSD_20080208_18-00-00.json",	"USDJPY_20141205_13-00-00.json",
            "GBPCHF_20020215_07-00-00.json",	'XAGUSD_20111220_14-00-00.json',
            "GBPCHF_20040504_09-00-00.json",    "XAUUSD_20021016_05-00-00.json",
            "GBPCHF_20140103_11-00-00.json",	'XAUUSD_20071010_03-00-00.json',
            "GBPJPY_20030707_09-00-00.json",	"XAUUSD_20080623_03-00-00.json",
            "GBPJPY_20060321_04-00-00.json",	"XAUUSD_20090903_05-00-00.json",
            "GBPJPY_20060727_00-00-00.json",	"XAUUSD_20120214_18-00-00.json",
            "GBPJPY_20071120_13-00-00.json",	"XAUUSD_20130412_19-00-00.json"    ]

def get_json(f):
    with open('jsons/' + f) as data_file:
        try:
            return json.load(data_file)
        except:
            print "None for", f
            pass

def check_json(f):
    with open('jsons/' + f) as data_file:
        try:
            j = json.load(data_file)
            return 1
        except:
            print "None for", f
            return 0

# some of the files don't load correctly...?
# easy check before running spark
working_files = [ f for f in files if check_json(f) ]


docs = sc.parallelize(working_files)
data = docs.map(lambda f: get_json(f))

# creating a tuple of (X, y)
Xy_data = data.map(lambda line: (line['vector'], line['class']))

# split data into train and test
train, test = Xy_data.randomSplit([0.8, 0.2], seed=0)

# Prep for modeling
X_train, y_train = train.map(lambda line: line[0]).collect(), train.map(lambda line: line[1]).collect()
X_test, y_test = test.map(lambda line: line[0]).collect(), test.map(lambda line: line[1]).collect()
















#
