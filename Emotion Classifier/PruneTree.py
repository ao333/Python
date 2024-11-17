def PruneTree(tree, maxDepth):

    def PruneTreeRecursive(node, height):
        if (height == maxDepth):
            node.kids = []
            node.op = ''
            return
        for child in node.kids:
            PruneTreeRecursive(child, height + 1)

    PruneTreeRecursive(tree, 0)
