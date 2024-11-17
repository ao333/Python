
def GetTreeHeight(tree):
    def GetTreeHeightRecursive(node, height):
        if len(node.kids) == 0:
            return height

        maxHeight = GetTreeHeightRecursive(node.kids[0], height + 1)
        for i in range(1, len(node.kids)):
            maxHeight = max(maxHeight,
                GetTreeHeightRecursive(node.kids[i], height + 1))
        return maxHeight
    return GetTreeHeightRecursive(tree, 0)
