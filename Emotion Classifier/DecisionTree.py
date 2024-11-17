from TreeNode import *
from MajorityValue import *
from copy import *

# function to parse data from matlab to usable array
def parse_data(data):
    return data['x'], data['y']

def Decision_Tree_Learning(examples, attributes, binary_targets):
    leaf = TreeNode()
    if all_targets_same(binary_targets):
        leaf.set_label(binary_targets[0][0])
        leaf.sign = binary_targets[0][0]
        return leaf
    if len(attributes) == 0:
        leaf.set_label(MajorityValue(binary_targets))
        leaf.sign = MajorityValue(binary_targets)
        return leaf

    best_attribute =  choose_best_decision_attribute(examples, attributes, binary_targets)
    leaf.op = best_attribute
    attributes.remove(best_attribute)
    newattribute_n = []
    newattribute_p = []
    for attribute in attributes:
        if attribute != best_attribute:
            newattribute_p.append(attribute)
            newattribute_n.append(attribute)
    newattribute_p = deepcopy(attributes)
    newattribute_n = deepcopy(attributes)
    # join columns tgt
    concat = np.column_stack((examples, binary_targets))

    # sort and partition into positive and negative for the best attribute
    p_stack, n_stack = split_stack(concat[concat[:, best_attribute].argsort()], best_attribute)

    # split columns between examples and binary targets for p_stack
    examples_p, binary_p = split_column(p_stack)
    # split columns between examples and binary targets for n stack
    examples_n, binary_n = split_column(n_stack)
    if len(examples_p) == 0 or len(examples_n) == 0:
        leaf.set_label(MajorityValue(binary_targets))
    else:
        p_subtree = Decision_Tree_Learning(examples_p, newattribute_p, binary_p)
        p_subtree.sign = 1
        leaf.add_child(p_subtree)
        n_subtree = Decision_Tree_Learning(examples_n,newattribute_n, binary_n)
        n_subtree.sign = 0
        leaf.add_child(n_subtree)

    return leaf

# function to check if all targets are same
def all_targets_same(binary_targets):
    return np.all(binary_targets == binary_targets[0,:], axis = 0)[0]

#function to that returns the best attribute based on information gain
def choose_best_decision_attribute(examples,attributes,binary_targets):
    # print('Binary Target Counts: ')
    p_count, n_count  = counts(binary_targets)
    initial_entropy = Entropy(p_count,n_count)
    best_attribute = -1
    best_info = -1000000
    for attribute in attributes:
        concat = np.column_stack((examples[:,attribute],binary_targets))

        p_examples,n_examples = split_stack(concat[concat[:, 0].argsort()],0)

        if len(n_examples) != 0:
            po, no = counts(n_examples[:, -1])
        else:
            po = 0
            no = 0
        if len(p_examples) != 0:
            p1, n1 = counts(p_examples[:, -1])
        else:
            p1 = 0
            n1 = 0
        total_count  = p_count + n_count
        info_gain = initial_entropy - ((po + no)/float(total_count)*Entropy(po,no) + (p1 + n1)/float(total_count)*Entropy(p1,n1))
        if info_gain > best_info:
            best_info = info_gain
            best_attribute = attribute

    return best_attribute

#function to count states
def counts(y):
    num_ones = np.count_nonzero(y)
    num_zeroes = len(y) - num_ones
    return num_ones, num_zeroes

#function to calculate Entropy
def Entropy(p, n):
    if p == 0 or n == 0:
        return 0
    total  = p + n
    return -p/float(total)*np.log2(p/float(total)) - n/float(total)*np.log2(n/float(total))

# function to split column
def split_column(stack):
    if len(stack) == 0:
        return [],[]
    return stack[:,:-1], stack[:,-1:]

def split_stack(stack, best_attribute):
    count = 0
    n_stack = []
    p_stack = stack
    for example in stack:
        if example[best_attribute] == 0:
            count += 1
        else:
            n_stack = stack[:count,:]
            p_stack = stack[count:,:]
            break
    return p_stack, n_stack
