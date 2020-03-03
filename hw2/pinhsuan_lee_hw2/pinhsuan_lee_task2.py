from pyspark import SparkContext
import sys
import time

def A_priori(data,support):   #support=threshold/NumPartition
    single_item = {}
    frequentitem = []
    basket=[]
    for i in data:
        basket.append(i)
        for item in i[1]:
            if item not in single_item:
                single_item[item] = 1
            else:
                single_item[item] += 1

    candidate = []
    for i in single_item:
        if single_item[i]>=support:
            candidate.append(i)
    frequentitem.extend(candidate) #frequent items list
    candidate_size = len(candidate)

    itemsize=2 #initial
    while candidate_size > itemsize:
        item_dic={}
        for i in range(0,candidate_size):
            for j in range(i+1,candidate_size):
                if itemsize == 2:
                    union_set=frozenset(set([candidate[i]]).union([candidate[j]]))
                else:
                    union_set=frozenset(set(candidate[i]).union(candidate[j]))
                if len(union_set) == itemsize:   #only count the items having one more item than previous state.
                    if union_set not in item_dic:
                        item_dic[union_set] = 1
                    else:
                        item_dic[union_set] += 1
        candi_waitforcheck=[]
        if itemsize == 2:
            for i in item_dic:
                if item_dic[i] >= 1:
                    candi_waitforcheck.append(i)
        else:
            for i in item_dic:
                if item_dic[i]>= itemsize:
                    candi_waitforcheck.append(i)
        nextRunList=[]
        for candi in candi_waitforcheck:
            count=0
            for dataset in basket:
                if candi.issubset(dataset[1]): #if candi is a subset of value, count+1
                    count+=1
                if count >= support:  #when it reaches the threshold, we add it to the frequent item list
                    fre_item=tuple(sorted(candi))
                    frequentitem.append(fre_item)
                    nextRunList.append(fre_item)
                    break
        candidate = nextRunList
        candidate_size = len(candidate)
        itemsize += 1
    return iter(frequentitem)

def count_FreqItem_P2(data,candidates):
    basket = []
    frequentItemsets = []
    for d in data:
        basket.append(d)
    for c in candidates:
        count = 0
        for b in basket:
            if type(c[0]) is tuple :
                if set(c[0]).issubset(b[1]):
                    count += 1
            elif type(c[0]) is str :        #single
                if set([c[0]]).issubset(b[1]):
                    count += 1
        if type(c[0]) is tuple:
            frequentItemsets.append((tuple(sorted(c[0])), count))
        else:    #single
            frequentItemsets.append((c[0], count))
    return iter(frequentItemsets)



def main():
    start_time=time.time()
    sc = SparkContext('local[*]', 'pinhsuan_lee_task2.py')
    allRDD=sc.textFile(sys.argv[3])

    header = allRDD.first()
    textRDD = allRDD.filter(lambda x: x != header)  # remove the header
    Filter_threshold=int(sys.argv[1])

    textRDD = textRDD.map(lambda x: x.split(',')).groupByKey().map(lambda x:(x[0],set(x[1]))).filter(lambda x:len(x[1]) > Filter_threshold)
    numPartition = allRDD.getNumPartitions()
    support=int(sys.argv[2])
    candidate = textRDD.mapPartitions(lambda x: A_priori(x, support / numPartition)).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).collect()  # get the possible frequent itemsets from each chunk
    Frequentset=textRDD.mapPartitions(lambda x:count_FreqItem_P2(x,candidate)).reduceByKey(lambda x,y:x+y).filter(lambda x:x[1] >= support).collect()

#deal with candidateList
    candi_single=[]
    candi_notsingle=[]
    for i in candidate:
        if type(i[0]) is str:
            candi_single.append(i[0])
        else:
            candi_notsingle.append(i[0])
    candi_single = sorted(candi_single)
    candi_notsingle = sorted(candi_notsingle)
    lenList_candi=[len(x)for x in candi_notsingle]
    maxLen_candi=max(lenList_candi) #find the max length in not single items(candidate)

#deal with frequent_itemList
    Fre_single=[]
    Fre_notsingle=[]
    for i in Frequentset:
        if type(i[0]) is str:
            Fre_single.append(i[0])
        else:
            Fre_notsingle.append(i[0])
    Fre_single=sorted(Fre_single)
    Fre_notsingle=sorted(Fre_notsingle)
    lenList_fre = [len(x) for x in Fre_notsingle]
    maxLen_fre = max(lenList_fre)  # find the max length in not single items(candidate)
#write to file
    output_task2 = sys.argv[4]


    with open(output_task2, "w") as output_file2:
        output_file2.write("Candidates:\n")
        for i in range(0,len(candi_single)-1):
            output_file2.write("('%s')," % candi_single[i])
        output_file2.write("('%s')\n\n" % candi_single[-1])
        for i in range(2,maxLen_candi+1):
            List=[m for m in candi_notsingle if len(m)==i]
            for k in range(0,len(List)-1):
                output_file2.write("{},".format(List[k]))
            output_file2.write("{}\n\n".format(List[-1]))

        output_file2.write("Frequent Itemsets:\n")
        for i in range(0,len(Fre_single)-1):
            output_file2.write("('%s')," % Fre_single[i])
        output_file2.write("('%s')\n\n" % Fre_single[-1])
        for i in range(2,maxLen_fre+1):
            List=[m for m in Fre_notsingle if len(m)==i]
            for k in range(0,len(List)-1):
                output_file2.write("{},".format(List[k]))
            output_file2.write("{}\n\n".format(List[-1]))
    output_file2.close()
    end_time=time.time()
    Duration=end_time - start_time
    print('Duration: %.2f'% Duration)

if __name__ == "__main__":
    main()
