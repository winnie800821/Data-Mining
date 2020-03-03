from pyspark import SparkContext
import sys
import time
import csv
from itertools import combinations
import random
# f(x)=(ax+b)%m



def prime_m(index):
    for i in range(index, index + 20):
        square_num = int(i ** 0.5)
        havefactorList = []
        for k in range(2, square_num + 1):
            if i % k != 0:
                havefactorList.append(0)
            else:
                havefactorList.append(1)
                break
        if sum(havefactorList) == 0:
            k = i
            break
    return k


def FindCandidate(data, columns, row):
    dictionary = {}
    candidates = []
    data = list(data)
    datastore = []
    for i in range(columns):
        siglist = []
        for j in range(row):
            datastore.append(data[j][i])
            siglist.append(data[j][i])
        siglist = tuple(siglist)
        if siglist not in dictionary:
            dictionary[siglist] = []
        dictionary[siglist].append(i)
    for k, v in dictionary.items():
        if len(v) > 1:
            candidates.extend(list(combinations(v, 2)))
    return iter(candidates)



def main():
    start_time=time.time()
    sc = SparkContext('local[*]', 'pinhsuan_lee_task1.py')
#    inputfile =sys.argv[1]
    inputfile='./yelp_train.csv'
    allRDD=sc.textFile(inputfile)
    header = allRDD.first()
    textRDD = allRDD.filter(lambda x: x != header).map(lambda x: x.split(','))  # remove the header
    business_user = textRDD.map(lambda x: (x[1],x[0])).groupByKey().mapValues(set).collect()
    userset=textRDD.map(lambda x:(x[0],1)).reduceByKey(lambda x, y: 1).map(lambda x: x[0]).collect()
    Num_user=len(userset) #rows
    Num_business=len(business_user) #columns


    dict_user={}
    for i in range(Num_user):
        if userset[i] not in dict_user:
            dict_user[userset[i]]=[]
        dict_user[userset[i]]=i
    dict_business = {}
    for i in range(Num_business):
        dict_business[i]=[]
        for u in business_user[i][1]:
            dict_business[i].append(dict_user[u])


    Num_hash = 165
    m = prime_m(Num_user)

    hashfunction = []
    random.seed(10000)
    for i in range(Num_hash):  # generate hash functions
        hashfunction.append([random.randint(0, 10000), random.randint(0, 10000)])
    # apply f(x)=(ax+b)%m


    signature_matrix = [[Num_user for col in range(Num_business)] for row in range(Num_hash)]
    for j in range(Num_business):
        for i in dict_business[j]:
            for k in range(Num_hash):
                signature_matrix[k][j] = min(signature_matrix[k][j],(i*hashfunction[k][0]+hashfunction[k][1]) % m)

    row = 3
    band = Num_hash / row
    LSH_RDD = sc.parallelize(signature_matrix, band)
    candidates = LSH_RDD.mapPartitions(lambda x: FindCandidate(x, Num_business, row)).map(lambda x: (x, 1)).reduceByKey(
        lambda x, y: 1).map(lambda x: x[0]).collect()
    Num_candidates = len(candidates)

    pair = []
    for i in candidates:
        m = []
        n = []
        for item in dict_business[i[0]]:
            m.append(item)
        for item in dict_business[i[1]]:
            n.append(item)
        len_m=len(m)
        len_n=len(n)
        m.extend(n)
        len_combine=len(set(m))
        union = len_combine
        intersection = len_m + len_n - len_combine
        Jac_similarity = intersection / union
        if Jac_similarity >= 0.5:
            if business_user[i[0]][0] < business_user[i[1]][0]:
                pair.append([(business_user[i[0]][0], business_user[i[1]][0]), Jac_similarity])
            else:
                pair.append([(business_user[i[1]][0], business_user[i[0]][0]), Jac_similarity])
    pair.sort()
#    output_task1=sys.argv[2]
    output_task1 = './output_task1.csv'
    with open(output_task1, "w") as output_task1:
        writer = csv.writer(output_task1)
        writer.writerow(['business_id_1', ' business_id_2', ' similarity'])

        for i in range(len(pair)):
            writer.writerow([pair[i][0][0], pair[i][0][1], pair[i][1]])

    end_time = time.time()
    total_time = end_time - start_time

    print('total time', total_time)


if __name__ == "__main__":
    main()