from pyspark import SparkContext
import sys
import time
import csv
from pyspark.mllib.recommendation import ALS
# f(x)=(ax+b)%m






def main():
    start_time=time.time()
    sc = SparkContext('local[*]', 'pinhsuan_lee_task2.py')
    '''
    inputfile=sys.argv[1]
    val_file=sys.argv[2]
    case = sys.argv[3]
    output_task2 = sys.argv[4]
    '''
    inputfile='./yelp_train.csv'
    val_file='./yelp_val.csv'
    case = '1'
    output_task2 = './task2_output1.csv'



    allRDD=sc.textFile(inputfile)
    header = allRDD.first()
    textRDD = allRDD.filter(lambda x: x != header).map(lambda x: x.strip('\n')).map(lambda x: x.split(','))  # remove the header
    businessset=textRDD.map(lambda x:(x[1],1)).reduceByKey(lambda x,y:1).map(lambda x:x[0]).collect()
    business_user = textRDD.map(lambda x: (x[1],(x[0],float(x[2])))).groupByKey().mapValues(set).collect()
    user_business=textRDD.map(lambda x:(x[0],x[1])).groupByKey().mapValues(set).collect()
    userset=textRDD.map(lambda x:(x[0],1)).reduceByKey(lambda x, y: 1).map(lambda x: x[0]).collect()


    dict_user={}   #key:user name, value:user index
    for i in range(len(userset)):
        dict_user[userset[i]]=i
    dict_business={}
    list_user=[k for (k,v) in dict_user.items()]
    for i in range(len(businessset)):  #key:business name, value:business index
        dict_business[businessset[i]]=i

    # testdata
    test_allRDD = sc.textFile(val_file)
    test_header = test_allRDD.first()
    testRDD = test_allRDD.filter(lambda x: x != test_header).map(lambda x: x.split(','))
    user_test = testRDD.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: 1).map(lambda x: x[0]).collect()
    bus_test = testRDD.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: 1).map(lambda x: x[0]).collect()

    for k in user_test:#add the user name in test data into dict
        if k not in dict_user:
            a = len(dict_user)
            dict_user[k] = a
    for k in bus_test:#add the bus name in test data into dict
        if k not in dict_business:
            c = len(dict_business)
            dict_business[k] = c
    userset=[k for k ,v in dict_user.items()]
    businessset=[k for k ,v in dict_business.items()]


    Num_user=len(dict_user) #rows
    Num_business=len(dict_business) #columns

    dict_business_user = {}   #key:business index, value:user index
    dic_matrix = {}     #key:(user index,business index), value:score

    dict_user_business = {}  # key:user index, value:business index
    for i in range(len(user_business)):
        dict_user_business[dict_user[user_business[i][0]]] = []
        for j in user_business[i][1]:
            dict_user_business[dict_user[user_business[i][0]]].append(dict_business[j])

    for i in range(len(business_user)):
        c = list(business_user[i][1])
        dict_business_user[i] = []
        for u in c:  # u=(user name,score)
            dict_business_user[i].append(dict_user[u[0]])
            dic_matrix[(dict_user[u[0]], i)] = u[1]
#testdata

    testdata = testRDD.map(lambda x: ((dict_user[x[0]], dict_business[x[1]]), 1)).reduceByKey(lambda x, y: 1).map(lambda x: x[0])  # user_bisiness
    testdata_result = testRDD.map(lambda x: ((dict_user[x[0]], dict_business[x[1]]), float(x[2]))).collect()  # [(user,bisiness),score]
    testdatalist = sorted(testdata.collect()) # (user,business)

    bus_user_test=testRDD.map(lambda x:(dict_business[x[1]],dict_user[x[0]])).groupByKey().mapValues(set).collect()
    user_business_test=testRDD.map(lambda x:(dict_user[x[0]],dict_business[x[1]])).groupByKey().mapValues(set).collect()
    dict_testdata = dict(testdata_result)



    if case== '1':
        rank = 10
        numIterations = 10
        user_bus_score = textRDD.map(lambda x: (dict_user[x[0]],dict_business[x[1]], x[2]))
        model = ALS.train(user_bus_score, rank, numIterations,  0.1, seed=200000)
        predictions = model.predictAll(testdata).map(lambda x: ((x[0], x[1]), x[2])).collect()
        dict_pred=dict(predictions)
        diff=[]
        for x in testdatalist:
            if x not in dict_pred:
                diff.append(x)
        for i in diff:
            predictions.append([i,3])

    if case== '3' :# Item-based
        predictions=[]
        for i in testdatalist:
            w={}
            total=0
            if i[0] in dict_user_business:
                for item in list(dict_user_business[i[0]]):
                    total+=dic_matrix[(i[0],item)]
                iav=total/len(list(dict_user_business[i[0]]))

            if i[0] in list(dict_user_business) and i[1] in list(dict_business_user):#old user,old item
                sum5=0
                sum6=0
                for m in set(dict_business_user[i[1]]):                #m is the user who rated the item
                    List = list(set(dict_user_business[m]).intersection(set(dict_user_business[i[0]]))) #corated item between user m and our predict user
                    number=len(List)
                    if number > 0:
                        sum1 = 0
                        sum2 = 0
                        for item in List:
                            sum1 += dic_matrix[(i[0], item)]
                            sum2 += dic_matrix[(m, item)]
                        av1 = sum1 / number
                        av2 = sum2 / number
                    numerator = 0
                    sum3 = 0
                    sum4 = 0
                    for r in List:
                        numerator+=(dic_matrix[(i[0],r)]-av1)*(dic_matrix[(m,r)]-av2)
                        sum3+=(dic_matrix[(i[0],r)]-av1)**2
                        sum4+=(dic_matrix[(m,r)]-av2)**2
                    if sum3==0 or sum4==0:
                        w[(i[0], m)]=0
                    else:
                        w[(i[0],m)]=numerator/((sum3**(1/2))*(sum4**(1/2)))
                    for r in List:
                        sum5+=(dic_matrix[(m,r)]-av2)*w[(i[0],m)]
                        sum6+=abs(w[(i[0],m)])
                if sum6==0:
                    predictions.append([(i[0], i[1]),iav])
                else:
                    predictions.append([(i[0], i[1]), iav+(sum5/sum6)])
            elif i[0] in list(dict_user_business) and i[1] not in list(dict_business_user):#old user,new item
                predictions.append([(i[0], i[1]), iav])
            elif i[0] not in list(dict_user_business) and i[1] in list(dict_business_user):
                k=list(dict_business_user[i[1]])
                add=0
                num=len(k)
                for m in k:
                    add+=dic_matrix[(m,i[1])]
                predictions.append([(i[0], i[1]), iav+add/num])
            elif i[0] not in list(dict_user_business) and i[1] not in list(dict_business_user):
                predictions.append([(i[0], i[1]), 3])


    if case == '2':  #User_based
        predictions=[]
        corated={}
        for k in testdatalist:
            w = {}
            if k[1] in dict_business_user and k[0] in dict_user_business:# if k[1] is not a new item
                for m in dict_user_business[k[0]]: #the business that user rated
                    List=list(set(dict_business_user[k[1]]).intersection(set(dict_business_user[m]))) #List of user who rated the two
                    size=len(List)
                    if size>1:
                        sumk = 0
                        summ = 0
                        for u in List: #u is user that rated m and k[1]
                           sumk+=dic_matrix[(u,k[1])]
                           summ+=dic_matrix[(u,m)]
                        corated_av1=sumk/size
                        corated_av2 = summ / size
                        corated[(k[1],m)]=(corated_av1 , corated_av2)
                        numerator=0
                        sum3=0
                        sum4=0

                        for u in List:
                            a=dic_matrix[(u,k[1] )]-corated[(k[1],m)][0]
                            b=dic_matrix[(u,m)]-corated[(k[1],m)][1]
                            numerator+=a*b
                            sum3+=a**2
                            sum4+=b**2
                        if sum3!=0 and sum4 !=0:
                            w[(k[1],m)]=numerator/((sum3**0.5)*(sum4**0.5))
                        else:
                            w[(k[1], m)] = 0
                    else:
                        w[(k[1], m)] = 0

                wList = [k for k, v in w.items() if v >= 0]
                numerator=0
                denominator=0
                n = 1
#                for c in wList[0:n]: #c=(k[1],m)

                for c in wList[0:n]: #c=(k[1],m)
                    List = list(set(dict_business_user[c[0]]).intersection(set(dict_business_user[c[1]]))) #corated item between user m and our predict user
                    for m in List:
                        numerator+=dic_matrix[(m,c[1])]*w[c]
                        denominator+=abs(w[c])
                if denominator==0:
                    rrr=0
                    for l in dict_user_business[k[0]]:
                        rrr+=dic_matrix[(k[0],l)]
                    cal=rrr/len(dict_user_business[k[0]])
                    predictions.append([k, cal])
                else:
                    a=numerator/denominator
                    predictions.append([k,a])
            elif k[1] not in dict_business_user and k[0] in dict_user_business: #if business it is a new  one, use the user avg rate
                total=0
                for m in dict_user_business[k[0]]:
                    total+=dic_matrix[(k[0],m)]
                r=total/len(dict_user_business[k[0]])
                predictions.append([k,r])
            elif k[1] in dict_business_user and k[0] not in dict_user_business:
                sumup=0
                for q in dict_business_user[k[1]]:
                    sumup+=dic_matrix[(q,k[1])]
                predictions.append([k,sumup/len(dict_business_user[k[1]])])
            elif k[1] not in dict_business_user and k[0] not in dict_user_business:
                predictions.append([k,3])



    dict_predict=dict(predictions)
    '''
    k=[]
    for i in testdatalist:
        k.append(abs(dict_predict[i] - dict_testdata[i]))
    sum1=0
    sum2=0
    sum3=0
    sum4=0
    sum5=0

    for v in k:
        if v>=0 and v<1:
            sum1+=1
        elif v>=1 and v<2:
            sum2+=1
        elif v>=2 and v<3:
            sum3+=1
        elif v>=3 and v<4:
            sum4+=1
        else:
            sum5+=1

    print('len(list1)',sum1)
    print('len(list2)',sum2)
    print('len(list3)',sum3)
    print('len(list4)',sum4)
    print('len(list5)',sum5)
    '''

    sigma = 0
    n = len(testdatalist)
    for i in testdatalist:
        sigma += (dict_predict[i] - dict_testdata[i]) ** 2

    RMSE = (sigma / n) ** (0.5)
    print("Mean Squared Error = %4f" % RMSE)


    with open(output_task2, "w") as output_task2:
        writer = csv.writer(output_task2)
        writer.writerow(['user_id', ' business_id', ' prediction'])
        for i in testdatalist:
            data=[userset[i[0]], businessset[i[1]], dict_predict[i]]
            writer.writerow(data)

    end_time = time.time()
    total_time = end_time - start_time
    print('total time',total_time)

if __name__ == "__main__":
    main()