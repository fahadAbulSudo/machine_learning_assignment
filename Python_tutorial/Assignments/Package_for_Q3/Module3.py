def intersecting_characters(str1, str2):
    #str1 = set(str1)
    #str2 = set(str2)
    str1 = set(str1)
    str2 = set(str2)
    result = str1.intersection(str2)
    return ",".join(result)

def freq(str1):
    str1 = str1.split(" ")
    dic = {}
    for i in str1:
        if i not in dic.keys(): #this condition use to check whether the key has 
            dic[i] = 0           #been used or not
        dic[i] = dic[i] + 1


    return dic


def merge(str1):
    new_str1 = str1[0:2]+str1[len(str1)-2:]
    return new_str1


def hyphen(str1):
    str_split = str1.split("-")
    str_sorted = sorted(str_split)
    result = "-".join(str_sorted)
    return result