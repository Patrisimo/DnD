import logging
import numpy as np

#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.CRITICAL)

class QuadTree:
  class RoadNode:
    def __init__(self, road, coords):
      self.road = road
      self.coords = coords
      self.id = road.id
    
    
 
  # This is a quadtree where a node either has data or children
  # A leaf will have at most 20 datapoints
  maxSize = 10
  
  order = [2, 1, 3, 0]
  roadNodes = {}
  # Children go
  # 1 0
  # 2 3
  
  def __init__(self, lowerLeft, upperRight, parent=None, name=None):
    self.root = self
    self.parent = parent
    self.isLeaf = True
    self.children = []
    self.data = []
    self.roads = []
    maxX, maxY = upperRight
    minX, minY = lowerLeft
    assert maxX > minX
    assert maxY > minY
    self.minX = minX
    self.maxX = maxX
    self.minY = minY
    self.maxY = maxY
    self.middleX = (maxX + minX) / 2
    self.middleY = (maxY + minY) / 2
    self.pointCount = 0
    self.roadCount = 0
    self.name = 'R' if name is None else name

    
    
  def addPoint(self, point):
    # If the point lies outside of this node, then create a node above it
    logging.info("Adding %s to %s" % (str(point), str(self)))
    assert point.id > 0
    x,y = point.coord
    newMinX = None
    newMinY = None
    
    if x < self.minX:
      logging.info('Too far left')
      newMinX = self.minX - (self.maxX - self.minX)
      newMaxX = self.maxX
    elif x > self.maxX:
      logging.info('Too far right')
      newMinX = self.minX
      newMaxX = self.maxX + (self.maxX - self.minX)
        
    if y < self.minY:
      logging.info('Too far down')
      newMinY = self.minY - (self.maxY - self.minY)
      newMaxY = self.maxY
    elif y > self.maxY:
      logging.info('Too far up')
      newMinY = self.minY
      newMaxY = self.maxY + (self.maxY - self.minY)
    
    if newMinX is not None or newMinY is not None:
      logging.info("Doesn't fit")
      if newMinX is None:
        newMinX = self.minX - (self.maxX - self.minX)
        newMaxX = self.maxX
      if newMinY is None:
        newMinY = self.minY - (self.maxY - self.minY)
        newMaxY = self.maxY        
      newMe = QuadTree((self.minX, self.minY), (self.maxX, self.maxY), parent=self)
      newMe.children = self.children
      newMe.data = self.data
      newMe.roads = self.roads
      newMe.pointCount = self.pointCount
      newMe.roadCount = self.roadCount
      
      assert self.parent is None
      self.children = []
      self.data = []
      self.roads = []
      self.middleX = (newMaxX + newMinX) / 2
      self.middleY = (newMaxY + newMinY) / 2
      self.minX = newMinX
      self.maxX = newMaxX
      self.minY = newMinY
      self.maxY = newMaxY
      self.makeChildren()
      self.addChild(newMe)
      return self.addPoint(point)
      
      
    # The point belongs in this node, try adding it to the data
    if self.isLeaf:
      logging.info("Fits")
      assert point.id not in [p.id for p in self.data], (str(self), str(point), '; '.join(map(str, self.data)))
      self.data.append(point)
      self.pointCount += 1
      if len(self.data) > QuadTree.maxSize: # We're full
        logging.info("Overflow")
        self.makeChildren()
            # Goes
            # 1 3
            # 0 2
        self.data = []
      return True
    
    # The point belongs in this node, but needs to be given to a child
    logging.info("Pass along")
    result = self.whichChild(point).addPoint(point)
    if result:
      self.pointCount += 1
    return result
  
  def addRoad(self, road):
    logging.info('Adding %s to %s' % (str(road), str(self)))
    assert road.id > 0
    belongs = True
      # Need to actually make sure the road belongs in here
      # Start needs to be to the left of the right side, and end needs to be right of the left
    x1,y1 = road.start.coord
    x2,y2 = road.end.coord
    x1,x2 = sorted([x1,x2])
    y1,y2 = sorted([y1,y2])
    
    if x1 > self.maxX or x2 < self.minX or y1 > self.maxY or y2 < self.minY:
      belongs = False

    logging.info('Belongs? %s' % belongs)
    
    if belongs:
      self.roadCount += 1
      self.roads.append(road)
      if self.isLeaf:
        QuadTree.roadNodes[road.id] = QuadTree.roadNodes.get(road.id, []) + [self]
      else:
        passThrough = self.whichChildrenRoad(road)
        logging.info([str(n) for n in passThrough])
        assert len(passThrough) > 0
        belongsBelow = False
        for c in passThrough:
          belongsBelow = c.addRoad(road) or belongsBelow
        assert belongsBelow
    return belongs
  def getRoadNodes(self, road):
    nodes = set()
    for n in QuadTree.roadNodes[road.id]:
      logging.info('Adding parent of %s' % (str(n)))
      nodes.add(n.parent)
    return nodes
  
  def getNode(self, point):
    if self.isLeaf:
      return self
    else:
      return self.whichChild(point).getNode(point)
  
  def getPoints(self, parents=0):
    if parents > 0 and self.parent is not None:
      return self.parent.getPoints(parents-1)
    else:
      if self.isLeaf:
        return self.data
      else:
        points = []
        for c in self.children:
          points.extend(c.getPoints())
        assert min([p.id for p in points], default=1) > 0, (str(self), '; '.join(map(str, points)))
        return points
  
  def getRoads(self, parents=0):
    if parents > 0 and self.parent is not None:
      return self.parent.getRoads(parents-1)
    elif self.isLeaf:
      return set(self.roads)
    else:
      roads = set()
      for c in self.children:
        roads = roads.union(c.getRoads())
      return roads
  
  def contains(self, point):
    n = self.getNode(point)
    ids = [p for p in n.data if p.id == point.id]
    return len(ids) > 0
  
  def makeChildren(self):
    assert self.parent is None or self.pointCount > 0
    self.isLeaf = False
    xVals = [self.minX, self.middleX, self.maxX]
    yVals = [self.minY, self.middleY, self.maxY]
    self.children = [None] * 4
    for i in range(2):
      for j in range(2):
        self.children[QuadTree.order[2*i+j]] =  QuadTree( (xVals[i], yVals[j]), (xVals[i+1], yVals[j+1]), parent=self, name=self.name+str(QuadTree.order[2*i+j]))
    for d in self.data:
      self.whichChild(d).addPoint(d)
    self.data = []
    for r in self.roads:
      QuadTree.roadNodes[r.id].remove(self)
      for c in self.whichChildrenRoad(r):
        c.addRoad(r)
    self.roads = []
  def addChild(self, child):
    x = int(child.middleX > self.middleX)
    y = int(child.middleY > self.middleY)
    self.children[QuadTree.order[2*x + y]] = child
    child.name += str(QuadTree.order[2*x+y])
    
    
    # Children are ABCD, with sides
    #   3 2
    # 4 B A 1
    # 5 C D 8
    #   6 7
    
  def whichChild(self, point):
    xIndex = int(point.coord[0] > self.middleX)
    yIndex = int(point.coord[1] > self.middleY)
    correct = self.children[QuadTree.order[2*xIndex + yIndex]]
    logging.info('%s belongs to %s' % (str(point), str(correct)))
    return correct
    # x,y = point.coord
    # assert (x - correct.minX ) * (x - )
  
  def whichChildrenRoad(self, road):
    # compute which of the four quadrants the road goes through
    children = [0,0,0,0]
    xCoords = [self.minX, self.middleX, self.maxX]
    yCoords = [self.minY, self.middleY, self.maxY]
    
    signs = np.array( [[aboveBelow(road, npa(x,y)) for y in yCoords] for x in xCoords])
    logging.info(np.transpose(signs, (1,0))[::-1])
    for i in range(0,3,2): # (0,0) or (2,2)
      start = (i,i)
      for j in range(0,3,2): # this is whether we are (2,0) or (0,2)
        end = (0+j,2-j)
        a,b = sorted([start,end], key=sum)
        
        if signs[a] *signs[b] <= 0: # Cross through this side
          midpoint = ( (a[0]+b[0])//2, (a[1]+b[1])//2)
          if signs[a] * signs[midpoint] <= 0: # Cross through the left half
            child = 2+i*(j-1)//2
            children[child] = 1
          if signs[b] * signs[midpoint] <= 0: # Cross through the right half
            child = (i-2)*(j-1)//2
            children[child] = 1
    # Only have a problem if 0 and 2 or 1 and 3 are in there
    assert sum(children) > 0
    #assert sum(children) < 3, (str(road), str(self),children)
    logging.info(children)
    logging.info(signs)
    if sum(children) == 2 and sum(i*j for i,j in zip(children,[1,0,1,0])) % 2 == 0:
      # Necessarily crosses the middle vertical
      if signs[1,0] * signs[1,1] <= 0: # Crosses through the lower half
        children[2] = 1
        children[3] = 1
      else: # Crosses through the upper half
        children[0] = 1
        children[1] = 1
      assert sum(children) == 3, children
    logging.info(children)
    return [ self.children[i] for i,n in enumerate(children) if n > 0]

  def removePoint(self, point):
    if self.isLeaf:
      assert point.id in [p.id for p in self.data]
      logging.info('Successfully removed %s from %s' % (str(point), str(self)))
      self.data = list(filter(lambda p: p.id != point.id, self.data))
      self.pointCount -= 1
    else:
      self.pointCount -= 1
      self.whichChild(point).removePoint(point)

  def removeRoad(self, road):
    for node in QuadTree.roadNodes[road.id]:
      node.roadCount -= 1
      assert road.id in [r.id for r in node.roads]
      node.roads = list(filter(lambda r: r.id != road.id, node.roads))
    QuadTree.roadNodes.pop(road.id)
    
  def sanityCheck(self):
    return True
    if self.isLeaf:
      assert len(self.children) == 0, (str(self), '; '.join(map(str, self.children)))
      assert self.pointCount == len(self.data), (str(self), ';'.join(map(str,self.data)))
      assert self.roadCount == len(self.roads), (str(self), ';'.join(map(str,self.roads)))
      assert min([p.id for p in self.data], default=1) > 0, (str(self), ';'.join(map(str,self.data)))
      assert min([r.id for r in self.roads], default=1) > 0, (str(self), ';'.join(map(str,self.roads)))
      logging.info('%s is good' % str(self))
    else:
      assert len(self.children) > 0
      for c in self.children:
        c.sanityCheck()
      assert self.pointCount == sum(c.pointCount for c in self.children), (str(self), '; '.join(map(str,self.children)), sum(c.pointCount for c in self.children))
      assert self.roadCount >= max([c.roadCount for c in self.children], default=0), (str(self), '; '.join(map(str,self.children)), sum(c.roadCount for c in self.children))
      logging.info('%s is good' % str(self))
    return True
  
  def __str__(self):
    return 'QuadTree#%s from (%.3f, %.3f) to (%.3f, %.3f) with%s children, %d points, and %d roads' % (self.name, self.minX, self.minY, self.maxX, self.maxY, 'out' if self.isLeaf else '', self.pointCount, self.roadCount)
  
def npa(a,b):
  return np.array([a,b])
    
def aboveBelow(road, pt): # positive is above
  return np.sign( np.linalg.det( np.vstack([road.start.coord - pt, road.end.coord-pt])))
