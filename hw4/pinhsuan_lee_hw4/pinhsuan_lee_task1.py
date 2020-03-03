from pyspark import SparkContext
import sys
from operator import add
from collections import deque
import time

def findBetweenness(userlist,dic_edge, last_betweenness):
    final_betweenness = {}
    for user in userlist:
        betweenness={}
        parent={}
        kid={}
        parent[user]=set()
        level=set([user])
        while len(level)>0:
            next_level=set()
            for vertex in level:
                temp=[]
                kid[vertex]=set(dic_edge[vertex]).difference(level).difference(parent[vertex])
                for par in dic_edge[vertex]:
                    if par not in parent:
                        parent[par]=set(dic_edge[par]).intersection(level)
                        temp.append(par)
                next_level.update(temp)
            level=next_level
        order=[node for node in parent][::-1]

        for node in order:
            nodeBetweenness = 1
            for child in kid[node]:
                if child > node:
                    key=(node,child)
                elif child < node:
                    key=(child,node)
                if key in betweenness:
                    nodeBetweenness += betweenness[key]
            NumParent=len(parent[node])
            for root in parent[node]:
                if root > node:
                    key = (node, root)
                elif root < node:
                    key = (root, node)
                betweenness[key]=nodeBetweenness/NumParent

        for edge in betweenness:
            if edge in final_betweenness:
                final_betweenness[edge]+= betweenness[edge]
            else:
                final_betweenness[edge] = betweenness[edge]

    final_betweenness=final_betweenness.items()
    return final_betweenness


def findCommunities(dic_edge,userlist,dict_community,communities):
    visit=set()
    communityNum=len(communities)
    for user in userlist:
        if user not in visit:
            NewCommunity = set()
            visit.add(user)
            queue=deque([user])
            while len(queue)>0:
                Vertex=queue[0]
                NewCommunity.add(Vertex)
                queue.popleft()
                if Vertex in dic_edge:
                    for child in dic_edge[Vertex]:
                        if child not in visit:
                            visit.add(child)
                            queue.append(child)
            if user in communities[dict_community[user]]:
                communities[dict_community[user]]=NewCommunity
            else:
                communities.append(NewCommunity)
                for user in NewCommunity:
                    dict_community[user]=communityNum
                communityNum+=1

def CalModularity(dict_edge, communities,m):
    modularity=0
    for community in communities:
        for i in community:
            Ki = len(dict_edge[i])
            for j in community:
                Kj = len(dict_edge[j])
                if j!=i and j in dict_edge[i]:
                    modularity+=1-(Ki*Kj)/(2*m)
                elif j!=i and j not in dict_edge[i]:
                    modularity-=(Ki*Kj)/(2*m)
    modularity /= (2 * m)
    return modularity




def Girvan_Newman(sc,dict_community,communities,edge,betweenness,m,userlist):
    max_Communities=communities.copy()
    NewEdge=edge.copy()
    max_modularity = CalModularity(edge, communities,m)
    while betweenness:
        removed_edge=betweenness[0][0]
        NewEdge[removed_edge[0]].remove(removed_edge[1])
        NewEdge[removed_edge[1]].remove(removed_edge[0])
        findCommunities(NewEdge,list(removed_edge),dict_community,communities)
        New_modularity=CalModularity(edge,communities,m)
        if New_modularity>max_modularity:
            max_modularity=New_modularity
            max_Communities=communities.copy()
        betweenness=sc.parallelize(userlist,1).mapPartitions(lambda x:findBetweenness(x,NewEdge,dict(betweenness[1:]))).reduceByKey(add).map(lambda x: ( x[0],x[1] / 2 )).collect()
        betweenness.sort(key=lambda x:(-x[1],x[0][0],x[0][1]),reverse=False)

    return max_Communities,max_modularity

def main():
    start_time=time.time()
    sc = SparkContext('local[*]', 'pinhsuan_lee_task1.py')


    threshold=int(sys.argv[1])
    inputfile =sys.argv[2]
    output_bet = sys.argv[3]
    output_community = sys.argv[4]
    '''
    threshold=7
    inputfile='./sample_data.csv'
    output_bet = './bet_output.txt'
    output_community = './com_output.txt'
    '''
    allRDD=sc.textFile(inputfile)
    header = allRDD.first()
    textRDD = allRDD.filter(lambda x: x != header).map(lambda x: x.split(','))  # remove the header
    user_business = textRDD.map(lambda x: (x[0],x[1])).groupByKey().mapValues(set).collect()
    alluser=textRDD.map(lambda x:(x[0],1)).reduceByKey(lambda x, y: 1).map(lambda x: x[0]).collect()
    Num_user=len(alluser) #rows

    edgetuple=[]
    dic_edge={}
    for i in range(Num_user):
        edge=[]
        for j in range(Num_user):
            intersec=list(set(user_business[i][1]).intersection(set(user_business[j][1])))
            if len(intersec) >= threshold and user_business[i][0]!= user_business[j][0]:
                edge.append(user_business[j][0])
#                if user_business[i][0] < user_business[j][0]:
#                    edgetuple.append((user_business[i][0], user_business[j][0]))
        if len(edge)>0:
            dic_edge[user_business[i][0]]=set(edge)
    userlist=[node for node in dic_edge]
    betweenness = sc.parallelize(userlist, 1).mapPartitions(lambda x: findBetweenness(x, dic_edge,{})).reduceByKey(add).map(lambda x: ( x[0],x[1] / 2 )).collect()
    betweenness.sort(key=lambda x:(-x[1],x[0][0],x[0][1]),reverse=False)



    with open(output_bet, "w") as output_file1:
        for k in range(len(betweenness)):
            output_file1.write("{}".format(betweenness[k][0]))
            output_file1.write(", {}\n".format(betweenness[k][1]))
    output_file1.close()

    m=len(betweenness)
    dict_community={edge:0 for edge in userlist}
    communities=[set(userlist)]

    findCommunities(dic_edge, userlist, dict_community, communities)
    MaxCommunity, Max_Modularity=Girvan_Newman(sc,dict_community,communities,dic_edge,betweenness,m,userlist)

    final_community = []
    for i in MaxCommunity:   #transfer set to list
        m=[]
        for element in i:
            m.append(element)
        m=sorted(m)
        final_community.append(m)
    final_community.sort(key=lambda x:(len(x),x[0]),reverse=False)


    with open(output_community, "w") as output_file2:
        for k in final_community:
            if len(k)==1:
                output_file2.write("'{}'\n".format(k[0]))
            else:
                for i in range(len(k)-1):
                    output_file2.write("'{}', ".format(k[i]))
                output_file2.write("'{}'\n".format(k[-1]))
    output_file2.close()
    end_time = time.time()
    total_time = end_time - start_time
    print('total time', total_time)


if __name__ == "__main__":
    main()