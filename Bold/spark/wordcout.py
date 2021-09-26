import sys
import re
from pyspark import SparkConf, SparkContext
conf = SparkConf()
sc = SparkContext(conf=conf)
lines=sc.textFile(sys.argv[1])
valid_chars=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# lowerd_case_lines=lines.flatMap(lambda line: line.lower())  /// I DON'T KNOW WHY THIS WAS GIVING ME EACH CHARACTER INSTEAD OF LOWERD CASE
# words_per_line=lines.flatMap(lambda l: l.split())
words_per_line = lines.flatMap(lambda l: re.split(r'[^\w]+', l))
actual_words=words_per_line.filter(lambda x: re.match(r'[a-zA-Z]',x))
# actual_wordsa=words_per_linea.filter(lambda x: re.match(r'[a-zA-Z]',x))
# actual_words=words_per_line.flatMap(lambda x: filter(lambda a: re.match(r'[a-z]',a), words_per_line))  //WHY IS THIS WRONG
lowerd_case_words=actual_words.map(lambda x:  (x.lower(),1)) #x[0] if x else   
count_wors=lowerd_case_words.map(lambda x: (x[0],x))
counts = lowerd_case_words.reduceByKey(lambda n1, n2: n1 + n2)
fina_cout=counts.map(lambda x: (x[0][0],1)).reduceByKey(lambda x,y: x+y)
final=fina_cout.sortByKey()
c=0
j=[0]*26
for x in valid_chars:
    for item in final.collect():
        if(x==item[0]):
            j[c]=item[1]
    c+=1
# print(j)
for i in range(0,26):
    print(valid_chars[i],'\t',j[i])
    

