list1 = [6,4,3,0,1]

#for ascending sort
for i in range(len(list1)-1):
    min_val = min(list1[i:])
    min_ind = list1.index(min_val, i)#Here "i" will help to take the value from that so duplicate vale also get sorted
    if list1[i] != list1[min_ind]:
        list1[i],list1[min_ind] = list1[min_ind],list1[i]

#for descending sort
for i in range(len(list1)):
    min_val = max(list1[i:])
    min_ind = list1.index(min_val)
    list1[i],list1[min_ind] = list1[min_ind],list1[i]

print(list1)