def add_node(v):
    if v in graph:
        print(v, "is already in the graph")
    else:
        graph[v] = []

def delete_node(v):
    if v not in graph:
        print(v, "is not present in the graph") 
    else:
        graph.pop(v)
        for i in graph:
            list1 = graph[i]
            if v in list1:
                list1.remove(v)

def delete_weighted_node(v):
    if v not in graph:
        print(v, "is not present in the graph") 
    else:
        graph.pop(v)
        for i in graph:
            list1 = graph[i]
            for j in list1:
                if v == j[0]:
                    list1.remove(j)
                    break

def add_undirected_edge(v1,v2,cost):
    if v1 not in graph:
        print(v1, "is not present in the graph")
    elif v2 not in graph:
        print(v2, "is not present in the graph")
    else:
        list1 = [v2,cost]
        list2 = [v1,cost]
        #graph[v1].append(v2)
        #graph[v2].append(v1)
        graph[v1].append(list1)
        graph[v2].append(list2)

def add_directed_edge(v1,v2,cost):
    if v1 not in graph:
        print(v1, "is not present in the graph")
    elif v2 not in graph:
        print(v2, "is not present in the graph")
    else:
        list1 = [v2,cost]
        #list2 = [v1,cost]
        graph[v1].append(list1)
        #graph[v2].append(list2)

def delete_undirected_edge(v1, v2):
    if v1 not in graph:
        print(v1, "is not present in the graph")
    elif v2 not in graph:
        print(v2, "is not present in the graph")
    else:
        if v2 in graph[v1]:
            graph[v1].remove(v2)
            graph[v2].remove(v1)

def delete_directed_edge(v1, v2):
    if v1 not in graph:
        print(v1, "is not present in the graph")
    elif v2 not in graph:
        print(v2, "is not present in the graph")
    else:
        if v2 in graph[v1]:
            graph[v1].remove(v2)
            #graph[v2].remove(v1)

def delete_weighted_edge(v1, v2, cost):
    if v1 not in graph:
        print(v1, "is not present in the graph")
    elif v2 not in graph:
        print(v2, "is not present in the graph")
    else:
        temp1 = [v1,cost]
        temp2 = [v2,cost]
        if temp1 in graph[v1]:
            graph[v1].remove(temp2)
            graph[v2].remove(temp1)

def DFS_weighted(node,visited,graph):
    if node not in graph:
        print("Node is not present in the graph")
        return
    if node not in visited:
        print(node)
        visited.add(node)
        for i in graph[node]:
            DFS_weighted(i[0],visited,graph)

visited = set()
graph = {}
add_node("A")
add_node("A")
add_node("B")
print(graph)