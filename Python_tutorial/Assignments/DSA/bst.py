class BST:
    def __init__(self,key):
        self.key = key
        self.lchild = None
        self.rchild = None

    def insert(self,data):
        if self.key is None:
            self.key = data
            return
        if self.key >= data:
            if self.lchild:
                self.lchild.insert(data)
            else:
                self.lchild = BST(data)
        else:
            if self.rchild:
                self.rchild.insert(data)
            else:
                self.rchild = BST(data)

    def search(self,data):
        if self.key == data:
            print("Node is found fuck off")
            return
        if data < self.key:
            if self.lchild:
                self.lchild.search(data)
            else:
                print("Not found fuck off")
        else:
            if self.rchild:
                self.rchild.search(data)
            else:
                print("Not found fuck off")

    def preorder(self):
        print(self.key)
        if self.lchild:
            self.lchild.preorder()
        if self.rchild:
            self.rchild.preorder()

    def delete(self,data):
        if self.key is None:
            print("Tree is empty")
            return
        if data < self.key:
            if self.lchild:
                self.lchild = self.lchild.delete(data)
            else:
                print("Given node is Not Present")
        elif data > self.key:
            if self.rchild:
                self.rchild = self.rchild.delete(data)
            else:
                print("Given Node is not present")
        else:
            if self.lchild is None:
                temp = self.rchild
                self = None
                return temp
            if self.rchild is None:
                temp = self.lchild
                self = None
                return temp
            node = self.rchild
            while node.lchild:
                node = node.lchild
            self.key = node.key
            self.rchild = self.rchild.delete(node.key)
        return self

def count(node):
    if node is None:
        return 0
    return 1+ count(node.lchild) + count(node.rchild)



root = BST(None)
root.insert(20)
if count(root)>1:
    root.delete(10)
else:
    print("can't delete tree")