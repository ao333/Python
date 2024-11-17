class TreeNode:
    def __init__(self, op='', label=''):
        self.op = op
        self.kids = []
        self.label = label
        self.sign = ''

    def add_child(self, child):
        self.kids.append(child)

    def set_label(self, value):
        self.label = value
