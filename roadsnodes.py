import logging
import numpy as np
from enum import Enum
import re
import copy
import random
from timer import Timer
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.CRITICAL)

class GoodType(Enum):
  EMPLOYMENT = 1 # demanded by industrial, produced by residential
  GOOD = 2 # demanded by residential, produced by commercial
  MATERIAL = 3 # demanded by commercial, produced by industrial
  CUSTOMER = 4 # demanded by commercial, produced by residential
  
  def roadDemand(good): # What type of road demands this?
    if good == GoodType.EMPLOYMENT:
      return IndustrialRoad
    if good == GoodType.GOOD:
      return ResidentialRoad
    if good == GoodType.MATERIAL:
      return CommercialRoad
    if good == GoodType.CUSTOMER:
      return CommercialRoad
  
  def roadSupply(good): # What type of road supplies this?
    if good == GoodType.EMPLOYMENT:
      return ResidentialRoad
    if good == GoodType.GOOD:
      return CommercialRoad
    if good == GoodType.MATERIAL:
      return IndustrialRoad
    if good == GoodType.CUSTOMER:
      return ResidentialRoad
  
  def get(n):
    if type(n) is int:
      return GoodType(n)
    if type(n) is GoodType:
      return n
    if type(n) is str:
      return GoodType(int(n))


class Terrain:
  terrainSet = {} # vector, threshold
  
  def __init__(self, nodes, extend=False): # A path of nodes that you aren't allowed to be to the right of
    self.nodes = [np.array(n) if type(n) is list else n for n in nodes]
    self.vectors = [ self.nodes[i] - self.nodes[i-1] for i in range(1, len(self.nodes))]
    self.vectors = [ np.array([-v[0], v[1]]) for v in self.vectors]
    self.thresholds = [ np.dot(n,v) for n,v in zip(self.nodes[1:], self.vectors)]
    
    

      
class Road:
  roadSet = {}
  roadId = 1
  recentlyAdded = []
  tree = None
  production = {}
  minLength = 10
  
  def getClass(name):
    if name == str(IndustrialRoad):
      return IndustrialRoad
    if name == str(TransportRoad):
      return TransportRoad
    if name == str(SpecialRoad):
      return SpecialRoad
    if name == str(CommercialRoad):
      return CommercialRoad
    if name == str(ResidentialRoad):
      return ResidentialRoad
    if name == str(CoastRoad):
      return CoastRoad
  def __init__(self, start, end, pop=0, level=1, id=0):
    logging.info('Creating road from %s to %s' % (str(start), str(end)))
    if start.coord[0] < end.coord[0]: # roads always go left to right
      self.start = start
      self.end = end
    elif start.coord[0] > end.coord[0]:
      self.start = end
      self.end = start
    else: # and down to up if vertical
      if start.coord[1] < end.coord[1]:
        self.start = start
        self.end = end
      else:
        self.start = end
        self.end = start
    # assert self.start != self.end, (str(start), str(end), str(self.start), str(self.end))
    assert not self.end.isBefore(self.start), (str(self.start), str(self.end))
    self.endpoints = (self.start.coord, self.end.coord)
    self.length = dist(self.start.coord, self.end.coord)
    #self.angle = angle(self.end.coord, self.start.coord) # measures angle with [1,0]
    self.level = level

    self.setMaxPop()
    self.supplies = {}
    self.production = {}
    self.storage = {}
    self.id = 0

      # Road.roadSet[id] = self
      # self.start.addRoad(self)
      # self.end.addRoad(self)

    self.transit = {}
    self.baseProduction = {}
    self.baseDemand = {}

    assert self.maxPop is not None, (self.level, self.length, str(self))
  def setMaxPop(self):
    raise Exception('setMaxPop not implemented for ' + self.__str__())
    
  def produce(self): # returns a dictionary: good -> level -> amt
    goods = {good: [self.maxPop * multiplier for i in range(self.level)]  for good, multiplier in self.goodsProduced.items()}
    self.production = goods
    self.baseProduction = {k:v for k,v in goods.items()}
    return goods
  
  def demand(self):
    demands = {}
    assert min(len(v) for v in self.supplies.values()) == self.level, (str(self), self.supplies)
    return self.supplies
    for good, demanded in self.supplies.items():
      i = 0
      while i < len(demanded) and demanded[i] == 0:
        i += 1
      if i == len(demanded): 
        demands[good] = (0, 0)
      else:
        demands[good] = (i, demanded[i])
    return demands
    
  def transport(self, amt):
    raise Exception('transport not implemented for ' + self.__str__())
    
  def travelTime(self):
    raise Exception('travelTime not implemented for ' + self.__str__())
  
  def color(self):
    raise Exception('color not implemented for ' + self.__str__())
  
  def otherEndpoint(self, node):
    if type(node) is Node:
      return self.start if self.end.id == node.id else self.end
    elif type(node) is int:
      return self.start.id if self.end.id == node else self.end.id

  def remove(self):
    logging.info('Removing %s' % str(self))
    if self.id != 0:
      Road.tree.removeRoad(self)
      Road.roadSet.pop(self.id)
      self.start.removeRoad(self)
      if self in self.end.roads: # in case start == end
        self.end.removeRoad(self) 
      self.id = 0
    else:
      raise Warning("Road already removed")

  def add(self, id=None):
    if self.id == 0:
      assert not np.isclose(self.length, 0)
      assert self.start != self.end
      self.id = Road.roadId if id is None else id
      Road.roadId += 1
      Road.roadSet[self.id] = self
      logging.info('Adding %s' % str(self))
      self.start.addRoad(self)
      assert self.id not in {r.id for r in self.end.roads}
      self.end.addRoad(self)
      self.reset()
      Road.recentlyAdded.append(self)
      Road.tree.addRoad(self)
    else:
      raise Warning("Road already added")
  
  def isActive(self):
    return self.id != 0
  
  def reset(self):
    self.supplies = {good: [self.maxPop * multiplier*((self.level - i)**2) for i in range(self.level)] for good, multiplier in self.goodsDemanded.items()}
    self.baseDemand = copy.deepcopy(self.supplies)
    self.storage = {}
    self.produce()
    assert len(self.supplies) > 0, str(self)
    assert min(len(v) for v in self.supplies.values()) == self.level, (str(self), self.supplies)
  
  def setLevel(self):
    logging.info('Supplies on edge %s: %s' % (str(self), str(self.supplies)))
    
    
    for i in range(self.level):
      broken = False
      for supply in self.supplies.values():
        if supply[i] > 1e-5: # The first level that is not supplied is our new level
          level = i+1
          broken = True
          break
      if broken:
        break
    else:
      level = self.level + 1
    # if level >= max(len(supply) for supply in self.supplies.values()): # enough to level up
      # assert not self.maxPop is None, str(self)
      # self.pop = self.maxPop / 10.
    # else:
      # self.pop = self.maxPop - sum(supply[level] for supply in self.supplies.values())/len(self.supplies)
    
    logging.info('Levelling up %s to level %d' % (str(self), level))
    self.level = level
    # However, need to penalize for having mixed traffic
    total = sum(self.transit.values())
    relatedTraffic = sum(self.transit.get(g,0) for g in self.goodsProduced) + sum(self.transit.get(g,0) for g in self.goodsDemanded)
    if relatedTraffic * 2 < total: 
      # If the most common resource is more than (level) as common as the next, gotta switch
      goods = sorted(self.transit.items(), key=lambda x: x[1], reverse=True)
      if len(goods) == 1:
        goods.append( (0,0))
      roadType = None
      if goods[0][1] > goods[1][1] * level:
        roadType = GoodType.roadDemand(goods[0][0])
      elif relatedTraffic * 2 < total:
        pass
      elif level < 2:
        roadType = TransportRoad
      else:
        level = max(int(level-goods[0][1] / goods[1][1]), 1)
    
      if not roadType is None:
        logging.info('Changing type of %s to %s' % (str(self), str(roadType)))
        newRoad = roadType(self.start, self.end)
        self.remove()
        newRoad.add()
  def inherit(self, parent):
    pass
  
  def __str__(self):
    return '%s#%dLv%d: %s -> %s' % (re.findall('[a-zA-Z]+Road', str(type(self)))[0], self.id, self.level, str(self.start), str(self.end))
    
  def getAngle(self, start=None):
    if start is None:
      start = self.start
      end = self.end
    elif start.id == self.start.id:
      start = self.start
      end = self.end
    elif start.id == self.end.id:
      start = self.end
      end = self.start
    
    return angle(end, start)
    
  def levelUp(self, score):
    if random.random() > np.exp(min(1,score)):
      self.level += 1
  
  def levelDown(self):
    self.level -= 1
    assert self.level > 0, str(self)

  def clearRecents():
    Road.recentlyAdded = []
  
  def getRecents():
    return list(filter(lambda x: x.id > 0, Road.recentlyAdded))
    
class SpecialRoad(Road):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.goodsProduced = {}
    self.goodsDemanded = {}
    #self.maxPop = 0
    self.transport = 1
    self.travel = 0
    self.colorSet = 'purple'
    self.data = None
    self.id
    self.buildingSize = 10

  def setMaxPop(self):
    self.maxPop = self.level * self.length
  
  def transportMultiplier(self):
    return self.transport
    
  def travelTime(self):
    return self.travel * self.length
  
  def color(self):
    return self.colorSet
  
  def inherit(self, parent):
    self.goodsProduced = parent.goodsProduced
    self.goodsDemanded = parent.goodsDemanded
    self.transport = parent.transport
    self.travel = parent.travel
    self.colorSet = parent.colorSet
    
  
  def create(self, goodsProduced, goodsDemanded, transport, travel, color):
    goodsProduced = {GoodType.get(k): v for k,v in goodsProduced.items()}
    goodsDemanded = {GoodType.get(k): v for k,v in goodsDemanded.items()}
    self.goodsProduced = goodsProduced
    self.goodsDemanded = goodsDemanded
    self.transport = transport
    self.travel = travel
    self.colorSet = color
    goodsProduced = {k.value: v for k,v in goodsProduced.items()}
    goodsDemanded = {k.value: v for k,v in goodsDemanded.items()}
    self.data = {'goodsProduced': goodsProduced, 'goodsDemanded': goodsDemanded, 'transport': transport, 'travel': travel, 'color': color}
    logging.info('Created SpecialRoad with %s' % (str(self.data)))
  
  def setLevel(self):
    logging.info('Special roads do not level up, currently level %d' % self.level)
    pass
        
        
  def levelUp(self, score):
    logging.info('Special roads do not level up')
    pass
class TransportRoad(Road):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.goodsProduced = {}
    self.goodsDemanded = {GoodType.GOOD: 0.1, GoodType.EMPLOYMENT: 0.1, GoodType.MATERIAL: 0.1}
    self.buildingSize = 5
    
  def setMaxPop(self):
    self.maxPop = self.level * self.length
  
  def transportMultiplier(self):
    return 1
    
  def travelTime(self):
    return self.length / self.level
  
  def color(self):
    return 'black'
  
    
class ResidentialRoad(Road):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    #self.goodsProduced = {GoodType.EMPLOYMENT: 2, GoodType.CUSTOMER: 1}
    self.goodsProduced = {GoodType.EMPLOYMENT: 5}
    self.goodsDemanded = {GoodType.GOOD: 1}
    assert not self.maxPop is None, str(self)
    self.buildingSize = 8 # 50 meters
    
  def setMaxPop(self):
    self.maxPop = self.level * self.length
    
  def transportMultiplier(self):
    return 1. / ( self.level)
  
  def travelTime(self):
    return 2 * self.length / (1 + self.level)
  
  def color(self):
    return 'green'

    
class CommercialRoad(Road):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.goodsProduced = {GoodType.GOOD: 3}
    #self.goodsDemanded = {GoodType.MATERIAL: 1, GoodType.CUSTOMER: 3}
    self.goodsDemanded = {GoodType.MATERIAL: 1}
    self.buildingSize = 10 # 100 meters
    
  def setMaxPop(self):
    self.maxPop = self.level * self.length

    
  def transportMultiplier(self):
    return 2. / (2. + self.level)
  
  def travelTime(self):
    return self.length / (1 + self.level)
  
  def color(self):
    return 'blue'

class IndustrialRoad(Road):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.goodsProduced = {GoodType.MATERIAL: 1}
    self.goodsDemanded = {GoodType.EMPLOYMENT: 50}
    self.buildingSize = 15 # 150 meters
    
  def setMaxPop(self):
    self.maxPop = self.level * self.length

  
  def transportMultiplier(self):
    return 2. / (2 + self.level)
  
  def travelTime(self):
    return self.length / (1 + self.level)
  
  def color(self):
    return 'orange'
    
    
class Node:
  nodeSet = {}
  nodeId = 1
  recentlyAdded = []
  tree = None
  
  def __init__(self, coord, y=None, road=None, roads=None, id=0):
    if y is None:
      self.coord = np.array(coord)
    else:
      self.coord = np.array([coord, y])
    if road is None:
      self.bisector = None
    else:
      self.bisector = road
    if roads is None:
      self.roads = []
    else:
      self.roads = roads
    self.storage = {}
    self.id = 0
    if id != 0:
      assert id not in Node.nodeSet
      self.add(id)
    self.unlocked = True # Can I add roads to this node?
    
  def remove(self):
    logging.info('Removing %s' % str(self))
    if len(self.roads) > 2:
      raise Exception('removing intersection with %d roads' % (len(self.roads)))
    elif self.id == 0:
      raise Warning('Node already removed')
    elif len(self.roads) == 0:
      Node.tree.removePoint(self)
      Node.nodeSet.pop(self.id)
      self.id = 0
    else:
      Node.nodeSet.pop(self.id)
      self.id = 0
      raise Warning("Need to join the edges")
  
  def replace(self, other):
    assert self.id != other.id
    logging.info('Replacing %s with %s' % (str(self), str(other)))
    for road in self.roads:
      logging.info('Adjusting road %s' % str(road))
      
      
      if road.start == self:
        start = other
        end = road.end
      else:
        end = other
        start = road.start
      newRoad = type(road)(start, end)
      if not np.isclose(newRoad.length, 0):
        newRoad.add()
      road.remove()
      
    
    if self.id > 0:
      raise Warning()
      self.roads = []
      self.remove()

  def addRoad(self, road):
    logging.info('Adding road %s to %s' % (str(road), str(self)))
    assert road.id not in {r.id for r in self.roads}, [r.id for r in self.roads]
    self.roads.append(road)
    
  def removeRoad(self, road):
    logging.info('Removing road %s from %s' % (str(road), str(self)))
    assert road.id in {r.id for r in self.roads}, [r.id for r in self.roads]
    self.roads = list(filter(lambda x: x.id != road.id, self.roads))
    if len(self.roads) == 0:
      logging.info('Orphaned: %s' % str(self))
      self.remove()
  
  def add(self, id=None):
    if self.id == 0:
      self.id = Node.nodeId if id is None else id
      logging.info('Adding %s' % str(self))
      Node.nodeId += 1
      Node.nodeSet[self.id] = self
      assert not Node.tree.contains(self), str(self)
      Node.recentlyAdded.append(self)
      assert Node.tree.addPoint(self)
      assert len(Node.nodeSet) == Node.tree.root.pointCount
      Timer.start('Bisecting road')
      if not self.bisector is None:
        logging.info('Bisecting %s at %s' % (str(self.bisector), str(self)))
        assert abs(angle(self.bisector.start, self, self.bisector.end) - np.pi/2) - np.pi/2 < 1e-5, (str(self), str(self.bisector), angle(self.bisector.start, self, self.bisector.end))
        road1 = type(self.bisector)(self.bisector.start, self)
        road1.inherit(self.bisector)
        road2 = type(self.bisector)(self, self.bisector.end)
        road2.inherit(self.bisector)
        road1.add()
        road2.add()
        self.bisector.remove()
      Timer.stop('Bisecting road')
    else:
      raise Warning("Node already added")
  
  def isActive(self):
    return self.id != 0
  
  def __str__(self):
    return '%d!(%5f,%5f)' % (self.id, self.coord[0], self.coord[1])
  
  def isBefore(self, other):
    if self.coord[0] < other.coord[0]:
      return True
    elif self.coord[0] > other.coord[0]:
      return False
    else:
      if self.coord[1] < other.coord[1]:
        return True
      elif self.coord[1] > other.coord[1]:
        return False
      else:
        return False
        
  def clearRecents():
    Node.recentlyAdded = []
  
  def getRecents():
    return list(filter(lambda x: x.id > 0, Node.recentlyAdded))
  
  
  
def dist(a,b):
  if hasattr(a,'coord'):
    return np.sqrt(np.square(a.coord - b.coord).sum())
  else:
    return np.sqrt(np.square(a-b).sum())    
  

def angle(a,b,c=None): # computes the angle <ABC (between 0 and pi), if c is ommitted then computes counterclockwise angle with horizontal
  full = False
  nodes = []
  if c is None:
    if hasattr(a,'coord'):
      nodes = [a,b]
      a,b = a.coord, b.coord
    c = b + np.array([1,0])
    full = True
  elif hasattr(a,'coord'):
    nodes = [a,b,c]
    a,b,c = a.coord, b.coord, c.coord

  assert type(a) is np.ndarray
  assert type(b) is np.ndarray
  assert type(c) is np.ndarray
  
  side1 = a - b
  side2 = c - b
  assert dist(side1, 0) * dist(side2, 0) > 0, ('; '.join(map(str,nodes)), a,b,c)
  assert abs(np.dot(side1, side2)) < dist(side1,0) * dist(side2,0) + 1e-5, ('; '.join(map(str,nodes)), a,b,c)
  
  angle_ = np.arccos( min(1, max(-1, np.dot(side1, side2) / (dist(side1, 0) * dist(side2, 0)))))

  if not full:
    return angle_ 
  else:
    if side1[1] > 0:
      return angle_ 
    else:
      return 2*np.pi - angle_ 
    
def listAdd(a,b):
  c = [i+j for i,j in zip(a,b)]
  longer = a if len(a) > len(b) else b
  c.extend(longer[len(c):])
  return c

def listMinus(a,b):
  c = [i-j for i,j in zip(a,b)]
  if len(a) > len(b):
    c.extend(a[len(c):])
  else:
    c.extend([-n for n in b[len(c):]])
  return c
  
  
def listDot(a,b):
  c = [i*j for i,j in zip(a,b)]
  return sum(c) 
  
def listDivide(a,b):
  c = [i/j if j > 0 else np.sign(i) for i,j in zip(a,b)]
  if len(a) > len(b):
    c.extend([np.sign(i) for i in a[len(c):]])
  else:
    c.extend([0]*len(b[len(c):]))
  return c
  
def listMultiply(a,b):
  c = [i*j for i,j in zip(a,b)]
  c.extend([0] * (max(len(a), len(b)) - len(c)))
  return c
  
def closeGreater(a,b):
  return a > b and not (np.isclose(b,a, atol=1e-6) or np.isclose(a,b,atol=1e-6))
  