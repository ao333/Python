import copy

def DeepCopyTree(tree):

    def DeepCopyTreeRecursion(node):
        clone = copy.copy(node)
        if len(node.kids) == 0:
            return clone

        clone.kids = []
        for child in node.kids:
            clone.kids.append(DeepCopyTreeRecursion(child))
        return clone

    return DeepCopyTreeRecursion(tree)

def DeepCopyTreeList(treeList):
    newList = []
    for tree in treeList:
        newList.append(DeepCopyTree(tree))
    return newList