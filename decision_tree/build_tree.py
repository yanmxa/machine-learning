class Node:
    def __init__(self, col=-1, value=None, leaf=None, trueBranch=None, falseBranch=None):
        self.col = col
        self.value = value
        self.leaf = leaf
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
def uniqueCounts(rows, column):
    valueCount = {}
    for row in rows:
        value = row[column]
        if value not in valueCount:
            valueCount[value] = 0
        valueCount[value] += 1
    return valueCount
def entropy(rows, column=-1):
    from math import log
    log2 = lambda x : log(x)/log(2)
    valueCount = uniqueCounts(rows,column)
    ent = 0.0
    for value in valueCount.keys():
        p = float(valueCount[value]) / len(rows)
        ent -= p * log2(p)
    return ent
def divideSet(rows, column, value):
    splitFunction = None
    if isinstance(value, int) or isinstance(value, float):
        splitFunction = lambda row : row[column] >= value
    else:
        splitFunction = lambda row : row[column] == value
    set1 = [row for row in rows if splitFunction(row)]
    set2 = [row for row in rows if not splitFunction(row)]
    return [set1, set2]
def buildTree(rows, impurity=entropy):
    if len(rows) == 0:
        return Node()
    currentImpurity = impurity(rows, -1)
    bestGain, bestCriteria, bestSets = 0.0, None, None
    for col in range(0, len(rows[0]) - 1):
        colValues = set()
        for row in rows:
            colValues.add(row[col])
        for value in colValues:
            (set1, set2) = divideSet(rows, col, value)
            p = float(len(set1)) / len(rows)
            nextImpurity = (p * impurity(set1, -1)) + ((1 - p) * impurity(set2, -1))
            gain = currentImpurity - nextImpurity
            if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                bestGain, bestCriteria, bestSets = gain, (col, value), (set1, set2)
    if bestGain > 0:
        trueBranch = buildTree(rows=bestSets[0], impurity=impurity)
        falseBranch = buildTree(rows=bestSets[1], impurity=impurity)
        return Node(col=bestCriteria[0], value=bestCriteria[1], trueBranch=trueBranch, falseBranch=falseBranch)
    else:
        return Node(leaf=uniqueCounts(rows, column=-1))
def classify(observation, tree):
    if tree.leaf != None:
        return tree.leaf
    else:
        value = observation[tree.col]
        branch = None
        if isinstance(value, int) or isinstance(value, float):
            if value >= tree.value: branch = tree.trueBranch
            else: branch = tree.falseBranch
        else:
            if value == tree.value: branch = tree.trueBranch
            else: branch = tree.falseBranch
        return classify(observation, branch)

def getWidth(tree):
    if tree.trueBranch == None and tree.flaseBranch == None:
        return 1
    return getWidth(tree.tb) + getWidth(tree.flaseBranch)
def getHeight(tree):
    if tree.trueBranch == None and tree.flaseBranch == None:
        return 0
    return max(getHeight(tree.trueBranch), getHeight(tree.flaseBranch)) + 1

def drawNode(draw, node, x, y):
    if node.leaf == None:
        w1 = getWidth(node.fb) * 100
        w2 = getWidth(node.tb) * 100
        left = x - (w1+w2)/2
        right = x + (w1+w2)/2
        draw.text((x-20,y-20), str(node.col)+':'+str(node.value), (0,0,0))
        draw.line((x,y, left+w1/2,y+75), fill=(255,0,0))
        draw.line((x,y, right-w2/2,y+75), fill=(0,255,0))
        drawNode(draw, node.fb, left+w1/2, y+100)
        drawNode(draw, node.tb, right-w2/2, y+100)
    else:
        for item in node.leaf.items():
            print('%s:%d'%item)
        txt=' \n'.join(['%s:%d'%v for v in node.leaf.items()])
        draw.text((x-20,y-20),txt,(0,0,0))

from PIL import Image, ImageDraw
def drawTree(tree, jpeg='tree.jpg'):
    w = getWidth(tree) * 100
    h = getHeight(tree) * 100 + 120
    image = Image.new('RGB',(w,h),(255,255,255))
    draw = ImageDraw.Draw(image)
    drawNode(draw, tree, w/2, 20)
    image.save(jpeg, 'JPEG')