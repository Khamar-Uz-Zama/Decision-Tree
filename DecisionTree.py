import xml.etree.cElementTree as ET
import numpy as np
import math
import pandas as pd
import copy
import argparse

class Node:
    
    def __init__(self, id, category, level, entropy, value=None, feature=None, text=None):
        
        self.id = id
        self.category = category
        self.entropy = entropy
        self.value = value
        self.feature = feature
        self.text = text
        self.level = level
        self.children = []
        self.parentID = -1
        
class Tree:
    
    nodeID = -1
    def createNode(self,category, level, entropy, value=None, feature=None, text=None):
        
        Tree.nodeID += 1
        return Node(Tree.nodeID, category, level, entropy, value, feature, text)
    
    def addNode(self, root, parentid, childNode):
        
        if(root.id == parentid):
            childNode.parentID = parentid
            root.children.append(childNode)
            return
        else:
            children = copy.copy(root.children)
            for node in children:
                if(node.id == parentid):
                    childNode.parentID = parentid
                    node.children.append(childNode)
                    return
                else:
                    if(len(node.children) != 0):
                        for n in node.children:
                            children.append(n)

def GetRequiredData(root, node, data):
    """
    Traverses from root to node and filters the data
    """
    currentNode = node
    constraints = {}
    dataFrame = pd.DataFrame(data)
    
    while(currentNode != None):
        constraints[currentNode.feature] = currentNode.value
        currentNode = GetParent(root, currentNode)
    
    for key, value in zip(constraints.keys(), constraints.values()):
        if(key == None and value == None):
            continue
        index = int(key[-1])
        dataFrame = dataFrame[dataFrame[index] == value]
           
    return dataFrame.to_numpy()

def GetRequiredNode(node):
    """
    Parses the tree from the node and returns the node which needs
    to be processed
    Uses Breadth First Search
    """
    if(len(node.children) == 0 and node.entropy!=0):
        return node
    else:
        allNodes = copy.copy(node.children)
        for element in allNodes:
            if(len(element.children) == 0 and element.entropy!=0):
                return element
            else:
                if(len(element.children) != 0):
                    for x in element.children: 
                        allNodes.append(x)
    return None

def GetParent(root, node):
    """
    Traverse the tree using root node
    Return the parent of node
    """    
    allNodes = copy.copy(root.children)
    if(root.id == node.id):
        return None
    else:
        for currentNode in allNodes:
            for child in currentNode.children:
                if(child.id == node.id):
                    return currentNode
                else:
                    allNodes.append(child)

    return None

def EntropyOfDataset(TargetColumn, NoOfClasses):
    """
    Requires only last column of dataset
    """
    TargetColumn = TargetColumn.reshape(TargetColumn.shape[0], 1)
    unique_values = np.unique(TargetColumn, return_counts=True)
    total_values = sum(i for i in unique_values[1])
        
    entropy = 0
    for x in unique_values[1]:
        ratio = x/total_values
        entropy += (-ratio)*math.log(ratio,NoOfClasses) 
        
    return entropy
        
def EntropyOfAttribute(Attr, TargetColumn, NoOfClasses):
    """
    1. Get all the unique values of the Attr
    2. Calculate Entropy for all unique values
    3. Calculate Average Information Gain
    """
    
    # 1. Get all the unique values of the Attr and separate them
    unique_attr_values = np.unique(Attr, return_counts=True)
    unique_targt_values = np.unique(TargetColumn, return_counts=True)
    
    SeparatedAttr = {}
    EntropyOfAttr = {}
    leafNodeText = {}
    
    for x in unique_attr_values[0]:
        SeparatedAttr[x] = []
        leafNodeText[x] = 0
        EntropyOfAttr[x] = 0
        
    for x in unique_attr_values[0]:
        for i in range(len(Attr)):
            if Attr[i] == x:
                SeparatedAttr[x].append(TargetColumn[i])
        
    # 2. Calculate Entropy for all unique values in the attrs           
    for x in unique_attr_values[0]:
        TargetCol = np.array(SeparatedAttr[x])
        temp = EntropyOfDataset(TargetCol, NoOfClasses)
        EntropyOfAttr[x] = temp
        if(temp == 0.0):
            leafNodeText[x]=SeparatedAttr[x][0]
        else:
            leafNodeText[x]=None
    
    # 3. Calculate Average Information Gain
    InformationGain = 0
    Total = unique_targt_values[1].sum()
    for attr, count in zip(unique_attr_values[0],unique_attr_values[1]):
        InformationGain += float(count/Total) * EntropyOfAttr[attr]
    
    return InformationGain, EntropyOfAttr, leafNodeText

def processDataset(data, Target, NoOfClasses):
    """
    Calculates the entropies of attributes and entropy of it's unique values
    """
    ent_of_attr = {}
    leafNodeText = {}
    indexes = {}
    ent_of_indiv_attr = {}

    for i in range(data.shape[1]):
        name = "att" + str(i)
        InformationGain, ent_of_indiv_attr[name], leafNode = EntropyOfAttribute(data[:,i], Target, NoOfClasses)
        entropy = EntropyOfDataset(Target, NoOfClasses)
        Gain = entropy - InformationGain
        ent_of_attr[name] = Gain
        leafNodeText[name] = leafNode
        indexes[name] = i
        
    return indexes, leafNodeText, ent_of_indiv_attr, ent_of_attr

def readFile(filePath):
    """
    Read the file and return as numpy array
    """
    csv_data = pd.read_csv(filePath, header = None)
    csv_data = np.array(csv_data)
    
    return csv_data

def createTree(csv_data):
    """
    Create a tree using the data
    Returns the root of the tree
    """
    Target = csv_data[:,-1]
    NoOfClasses = len(np.unique(Target))
    treeEntropy = EntropyOfDataset(Target, NoOfClasses)    
    tree = Tree()
    level = 0
    root = tree.createNode("tree",0,treeEntropy)

    while(True):
        currentNode = GetRequiredNode(root)
        if(currentNode == None):
            break
        data = GetRequiredData(root, currentNode, csv_data)
        Target = data[:,-1]
        data = data[:, :-1]
    
        indexes, leafNodeText, ent_of_indiv_attr, ent_of_attr = processDataset(data, Target, NoOfClasses)
        max_entropy_attr = max(ent_of_attr, key = ent_of_attr.get) 
        level += 1
        maxleafNodeText = leafNodeText[max_entropy_attr]
        for  item in ent_of_indiv_attr[max_entropy_attr].items():
            if(item[1]==0.0):
                text = maxleafNodeText[item[0]]
            else:
                text = None
            node = tree.createNode(category="node", level=level, entropy=item[1], value=item[0], feature=max_entropy_attr, text=text)
            
            tree.addNode(root,currentNode.id, node)
    
    return root

def createXML(root, xmlPath):
    """
    Create an xml file by traversing the root node
    """    
    
    rootElement = ET.Element(root.category, idm=str(root.id), entropy = str(root.entropy))
    parentElement = rootElement
    nodes = copy.copy(root.children)
    
    # Add elements in ascending order of the entropy
    nodes.sort(key= lambda x:x.entropy)

    for node in nodes:
        pNode = GetParent(root,node)
        if(pNode != None):
            predicate = ".//node[@value='"+pNode.value+"'][@idm='"+str(pNode.id)+"'][@entropy='"+str(pNode.entropy)+"'][@feature='"+pNode.feature+"']"                    
            parentElement = rootElement.find(predicate)
        temp = node.category + str(node.id)                                 
        if(node.entropy == 0):
            temp = ET.SubElement(parentElement, node.category,idm=str(node.id),entropy=str(node.entropy), feature=str(node.feature), value=str(node.value))
            temp.text = str(node.text)
        else:
            ET.SubElement(parentElement, node.category,idm=str(node.id), entropy=str(node.entropy), feature=str(node.feature), value=str(node.value))

        if(len(node.children)>0):
            temp = copy.copy(node.children)
            temp.sort(key= lambda x:x.entropy)

            for child in temp:
                nodes.append(child)
    
    tree = ET.ElementTree(rootElement)
    for x in rootElement.getiterator():
        x.attrib.pop("idm")
    tree.write(xmlPath)
        
    return

def main():
    """
    calls every function - does nothing
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', help='Data File')
    parser.add_argument('--output', help='Output File')
    
    args = parser.parse_args()
    dataPath = args.data
    xmlPath = args.output
    
    csv_data = readFile(dataPath)
    root = createTree(csv_data)
    createXML(root, xmlPath)

main()
