def add_node(v):
    global node_count
    if v in nodes:
        print(v, "is already present in the graph")
    else:
        node_count = node_count + 1
        nodes.append(v)
        for n in graph:
            n.append(0)
        
        temp = []
        for i in range(node_count):
            temp.append(0)
        graph.append(temp)

def delete_node(v):
    global node_count
    if v not in nodes:
        print(v, "is not present in the graph")
    else:
        index1 = nodes.index(v)
        node_count = node_count-1
        nodes.remove(v)
        graph.pop(index1)
        for i in graph:
            i.pop(index1)

def add_undirected_edge(v1,v2,data):
    if v1 not in nodes:
        print(v1, "is not present in the graph")
    elif v2 not in nodes:
        print(v2, "is not present in the graph")
    else:
        index1 = nodes.index(v1)
        index2 = nodes.index(v2)
        graph[index1][index2] = data
        graph[index2][index1] = data

def add_directed_edge(v1,v2,data):
    if v1 not in nodes:
        print(v1, "is not present in the graph")
    elif v2 not in nodes:
        print(v2, "is not present in the graph")
    else:
        index1 = nodes.index(v1)
        index2 = nodes.index(v2)
        graph[index1][index2] = data
        #graph[index2][index1] = data

def delete_edge(v1, v2):
    if v1 not in nodes:
        print(v1, "is not present in the graph")
    if v2 not in nodes:
        print(v2, "is not present in the graph")
    else:
        index1 = nodes.index(v1)
        index2 = nodes.index(v2)
        graph[index1][index2] = 0
        graph[index2][index1] = 0

def print_graph():
    for i in range(node_count):
        for j in range(node_count):
            print(format(graph[i][j],"<3"),end=" ")
        print()

nodes = []
graph = []
node_count = 0
add_node("A")
add_node("B")
print(graph)
print_graph()
print(nodes)