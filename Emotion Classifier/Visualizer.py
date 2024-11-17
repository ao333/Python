import os
from graphviz import Source

class Visualizer:
    __idx = 0
    def __init__(self, filename, tree):
        self.__filename = filename
        self.__root = tree

    def build(self, label = ""):
        (_, result) = self.build_node(self.__root)
        result = ['digraph Tree {\n', 'node [shape=box];\n', 'label="' + label + '";'] + result + ['}\n']

        text = ""
        for line in result:
            text += line + "\n"
        dot = Source(text)
        dot.format = 'png'
        dot.render(self.__filename)
        os.remove(self.__filename)

    def build_node(self, node):
        result = []
        idx = self.__idx;
        child_result = []
        child_links = []
        for child in node.kids:
            (child_idx, rresult) = self.build_node(child)
            child_result = child_result + rresult
            child_links.append((child_idx, child))
            idx = max(idx, child_idx)
        self.__idx = idx + 1
        if len(node.kids)==0:
            result.append(str(idx) + ' [shape=none, label="' + ('Yes' if node.sign == 1 else 'No') + '"];\n')
        else:
            result.append(str(idx) + ' [label="' + str(node.op) + '"];\n')
        result = result + child_result
        for (cidx, child) in child_links:
            if len(child.kids) == 0:
                result.append(str(idx) + ' -> ' + str(cidx - 1) + ' [headlabel="' + str(child.label) + '", labeldistance=2.5];\n')
            else:
                result.append(str(idx) + ' -> ' + str(cidx - 1) + ';\n')
        t = ((idx + 1), result)
        return t
