from pyspark import SparkContext
import sys
import json
import time


sc = SparkContext('local[*]', 'pinhsuan_lee_task2.py')
review = sys.argv[1]
business = sys.argv[2]
Re_textRDD=sc.textFile(review)
Re_textRDD=Re_textRDD.map(lambda line:line.split('\n')).map(lambda line:json.loads(line[0]))
Bus_textRDD=sc.textFile(business)
Bus_textRDD=Bus_textRDD.map(lambda line:line.split('\n')).map(lambda line:json.loads(line[0]))
count_avg_star=Re_textRDD.map(lambda x:(x["business_id"],x["stars"])).sortByKey(lambda a,b: (a[0]+b[0],a[1]+b[1]))
busID_state= Bus_textRDD.map(lambda a:(a["business_id"],a["state"]))

state_star=count_avg_star.join(busID_state)
compute=state_star.map(lambda x: (x[1][1],(x[1][0],1))).reduceByKey(lambda x,y:(x[0]+y[0],x[1]+y[1])).\
    sortByKey(True).map(lambda x:(x[0],x[1][0]/x[1][1])).map(lambda x:(x[1],x[0]))
avg_state_star=compute.sortByKey(False).collect()
print(avg_state_star)

output_task2_1=sys.argv[3]
with open(output_task2_1, "w") as output_file1:
    output_file1.write("state,stars\n")
    for info in avg_state_star:
        output_file1.write("{},{}\n".format(info[1], info[0]))

dic_time={}
#method 1
start_timer1=time.time()
method1_RDD=compute.sortByKey(False).collect()
print(method1_RDD[0:5])
end_timer1=time.time()
dic_time["m1"]=end_timer1-start_timer1
#method 2
start_timer2=time.time()
method2_RDD=compute.sortByKey(False).take(5)
print(method2_RDD)
end_timer2=time.time()
dic_time["m2"]=end_timer2-start_timer2

dic_time["explanation"]="In the first method, we collect all the pairs and compare to get the top 5 states."\
                        "However, in the second method, we compare and get the top 5 states locally."\
                        "Therefore, the first method takes more time than the second one."
print(dic_time)
output_task2_2=sys.argv[4]

with open(output_task2_2, "w") as output_file2:
    json.dump(dic_time, output_file2,indent=1)

