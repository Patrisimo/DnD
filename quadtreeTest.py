from quadtree import QuadTree
from roadsnodes import Node, TransportRoad

from PIL import Image, ImageDraw
import random
import itertools

QuadTree.maxSize = 1
qt = QuadTree( (-160,-160), (160,160))

assert qt.isLeaf
assert qt.pointCount == 0
print('Test 1 passed')
pts = [[Node(i+10*random.random(),j+10*random.random()) for j in range(-170,180,20)] for i in range(-170,180,20)]
for row in pts:
  for p in row:
    p.add()
qt.addPoint(pts[1][1])

assert qt.isLeaf
assert qt.pointCount == 1
print('Test 2 passed')
qt.addPoint(pts[6][2])

assert not qt.isLeaf
assert qt.pointCount == 2
print('Test 3 passed')
qt.addPoint(pts[0][0])

assert qt.minX < -16
assert qt.pointCount == 3
print('Test 4 passed')


def draw(tree, fname):
  img = Image.new('RGBA', (tree.maxX-tree.minX + 10, tree.maxY - tree.minY +10), (255,255,255,255))
  draw = ImageDraw.Draw(img)
  
  adjX = tree.minX - 5
  adjY = tree.maxY + 5
  
  for road in qt.roads:
    print('Coloring %s' % str(road))
    for node in qt.getRoadNodes(road):
      print('Filling in %s' % (str(node)))
      draw.rectangle([(node.minX - adjX, -node.minY + adjY), (node.maxX - adjX, -node.maxY + adjY)], fill='green')
  
  for road in qt.roads:
    print('Drawing %s' % str(road))
    draw.line([road.start.coord[0] - adjX, -road.start.coord[1] + adjY, road.end.coord[0] - adjX, -road.end.coord[1] + adjY], fill='yellow')
  
  toDraw = [tree]
  while len(toDraw) > 0:
    t = toDraw.pop(0)
    #print('Drawing %s' % str(t))
    draw.line( [(t.minX-adjX, -t.middleY+adjY), (t.maxX-adjX, -t.middleY+adjY)], fill='black')
    draw.line( [(t.middleX-adjX, -t.minY+adjY), (t.middleX-adjX, -t.maxY+adjY)], fill='black')
    for s in t.children:
      toDraw.append(s)
  
  toDraw = [tree]
  while len(toDraw) > 0:
    t = toDraw.pop()
    for s in t.children:
      toDraw.append(s)
    for d in t.data:
      x,y = d.coord
      draw.ellipse([(x-1-adjX,-y-1+adjY),(x+1-adjX,-y+1+adjY)], fill='red')
      
  
  img.save(fname)

  
  

draw(qt, 'tree1.png')
print('Tree1 complete')
qt = QuadTree( (-155, -155), (155,155))
points = [Node(x,y) for x,y in [(30,50),(30,70),(50,90),(70,90),(90,70),(90,50),(70,30),(50,30)]]
p1 = Node(50,50)
p1.add()
p2 = Node(50,70)
p2.add()
[p.add() for p in points]
pairs = [(a,b) for a in points for b in points if (a.coord-b.coord).sum() > 0]
roads = [TransportRoad(a,b) for a,b in pairs]
[r.add() for r in roads]
for i in range(len(roads)):
  qt = QuadTree( (-155,-155), (155,155))
  QuadTree.roadNodes = {}

  qt.addPoint(p1)

  qt.addPoint(p2)
  [qt.addPoint(p) for p in points]
  qt.addRoad(roads[i])
  draw(qt, 'treeroad%d.png' % i)
  print('Treeroad%d complete' % i)

QuadTree.roadNodes = {}
qt = QuadTree( (-160,-160), (160,160))
added = []
for i in range(100):
  
  #(row,col) in random.sample(list(itertools.product(range(1,len(pts)-1), range(1,len(pts)-1))), 100):
  x = (random.random()-.5) * 320
  y = (random.random()-.5) * 320
  p = Node(x,y)
  p.add()
  assert qt.parent is None
  qt.addPoint(p)
  added.append(p)
  draw(qt, 'tree2-%d.png' % i)



toCheck = [qt]
while len(toCheck) > 0:
  t = toCheck.pop()
  for c in t.children:
    toCheck.append(c)
  assert t.isLeaf or t.pointCount > 0, str(t)

print('Tree2 complete')

for i in range(20):
  QuadTree.roadNodes = {}
  qt = QuadTree( (-160,-160), (160,160))
  added = []
  for _ in range(100):
    x = (random.random()-.5) * 320
    y = (random.random()-.5) * 320
    p = Node(x,y)
    p.add()
    assert qt.parent is None
    qt.addPoint(p)
    added.append(p)
  a,b = random.sample(added,2)
  road = TransportRoad(a,b)
  road.add()

  qt.addRoad(road)

  draw(qt, 'treeroadr%d.png' % i)
  print('Treeroadr%d complete' % i)

