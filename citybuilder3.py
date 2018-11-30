import numpy as np
from itertools import chain
from roadsnodes import *
import random
from PIL import Image, ImageDraw, ImageColor
import logging
from argparse import ArgumentParser
import json
import copy
from timer import Timer
from quadtree import QuadTree

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.CRITICAL)
logging.disable(logging.CRITICAL)


def options():
  parser = ArgumentParser()
  
  parser.add_argument('iterations', type=int)
  parser.add_argument('--load')
  parser.add_argument('--draw-every', default=1, type=int)
  parser.add_argument('--no-save', action='store_true', default=False)
  parser.add_argument('-n', '--name', default='test')
  
  return parser.parse_args()

def main():
  ops = options()
  
  if ops.load is None:
    # FILENAME = 'newertest'
    # a = Node(0,0)
    # b = Node(40, 0)
    # c = Node(10, 10)
    # d = Node(100, 90)
    # a.add()
    # b.add()
    # c.add()
    # d.add()
    tree = QuadTree( (0,0), (1000,1000))
    Node.tree = tree
    Road.tree = tree
    FILENAME = ops.name
    a1 = Node(0,0)
    a2 = Node(10,0)
    a1.add()
    a1.unlocked = False
    a2.add()    
    a2.unlocked = False
    
    b1 = Node(790,0)
    b2 = Node(800,0)
    b1.add()
    b1.unlocked = False
    b2.add()
    b2.unlocked = False
    
    c1 = Node(400,380)
    c2 = Node(400,390)
    c1.add()
    c1.unlocked = False
    c2.add()
    c2.unlocked = False
    
    d1 = Node(380, 0)
    d2 = Node(420, 0)
    d3 = Node(400, -20)
    d4 = Node(400, 20)
    d1.add()
    d2.add()
    d3.add()
    d4.add()
    
    center = [d1, d3, d2, d4, d1]
    for i in range(1,5):
      r = ResidentialRoad(center[i], center[i-1])
      r.add()
    
    WestGate = SpecialRoad(a1, a2)
    WestGate.create({GoodType.MATERIAL: 40}, {GoodType.GOOD: 40}, 1, 1, 'purple')
    WestGate.add()
    EastGate = SpecialRoad(b1, b2)
    EastGate.create({GoodType.MATERIAL: 40}, {GoodType.GOOD: 40}, 1, 1, 'purple')
    EastGate.add()
    Dock = SpecialRoad(c1, c2)
    Dock.create({GoodType.MATERIAL: 80}, {GoodType.GOOD: 80}, 1, 1, 'navy')
    Dock.add()
    
    
    # assert a.isActive()
    
    
    # X = SpecialRoad(a,c)
    # X.create({GoodType.MATERIAL: 20}, {GoodType.GOOD: 20}, 1, 1, 'purple')
    # X.add()
    # Y = TransportRoad(c,d)
    # Y.add()
    
    # assert Y in d.roads
    startIter = 0
    drawCity('%s.png' % (FILENAME))
  else:
    FILENAME, startIter = load(ops.load) 
  try:
    for iterCount in range(startIter, startIter + ops.iterations):
      cycle()
      if iterCount % ops.draw_every == 0:
        Timer.start('Drawing')
        drawCity('%s%02d.png' % (FILENAME, iterCount), clean=True, clusters=False)
        drawCity('%s_clusters%02d.png' % (FILENAME, iterCount), clean=True, clusters=True)
        Timer.stop('Drawing')
  except KeyboardInterrupt:
    pass
  finally:
    print(Timer.report())
    print('Saving...')

    drawCity('%s_final.png' % (FILENAME), clean=True, clusters=False)
    drawCity('%s_clusters_final.png' % (FILENAME), clean=True, clusters=True)

    nodeInfo = [{'coord': n.coord.tolist(), 'id': n.id, 'roads': [r.id for r in n.roads]} for n in Node.nodeSet.values()]
    roadInfo = [{'start': r.start.id, 'end': r.end.id, 'type': str(type(r)), 'level': r.level, 'data': r.data if type(r) is SpecialRoad else None, 'id': r.id} for r in Road.roadSet.values()]
      
    with open('%s.json' % FILENAME, 'w') as file:
      json.dump({'nodes': nodeInfo, 'roads': roadInfo, 'filename': FILENAME, 'iter': iterCount, 'treeMin': Node.tree.getMin(), 'treeMax': Node.tree.getMax()}, file)

def load(filename):
  with open(filename) as file:
    data = json.load(file)
  FILENAME = data['filename']
  startIter = data['iter']
  tree = QuadTree( data['treeMin'], data['treeMax'])
  Node.tree = tree
  Road.tree = tree
  for node in data['nodes']:
    Node(node['coord'], id=node['id'])
  Node.nodeId = max(Node.nodeSet.keys())+1
  
  for road in data['roads']:
    newRoad = Road.getClass(road['type'])(Node.nodeSet[road['start']], Node.nodeSet[road['end']], level=road['level'])
    if type(newRoad) is SpecialRoad:
      newRoad.create(**road['data'])
    newRoad.add(road['id'])
    newRoad.reset()
  Road.roadId = max(Road.roadSet.keys())+1 
  return FILENAME, startIter
      
def cycle():
  # First, look for two nodes that are close together but not connected 
  # Say, take the pair with the smallest dist(a,b) / roadDist(a,b)
  
  Timer.start('Cleaning')
  assert Node.tree.sanityCheck()
  cleanNodes(Node.nodeSet)
  cleanRoads(Road.roadSet)
  split(Road.roadSet)
  Timer.stop('Cleaning')
  
  Timer.start('Connecting graph')
  closeRelativeDistance = 0
  attempts = 0
  while attempts < 10:
    attempts += 1
    (node1, node2), closeRelativeDistance = findClosePair(Node.nodeSet, Road.roadSet)
    if node1 is not None and closeRelativeDistance < 0.2:
      if not node1.isActive():
        node1.add()
      makeRoad(node1, node2, Road.roadSet, ResidentialRoad)
    else:
      break
  else:
    logging.info('***RAN OUT OF ATTEMPTS***')
  Road.clearRecents()
  Node.clearRecents()
  Timer.stop('Connecting graph')
  
  Timer.start('Cluster assignment')
  # Now assign clusters
  # First, connect transport roads
  clusterRoads, roadClusters = assignClusters(Node.nodeSet, Road.roadSet)
  Timer.stop('Cluster assignment')
  
  Timer.start('Production')
  # Now production
  # First, each road completes its production
  for road in Road.roadSet.values():
    road.reset()
  
  # All the roads that don't belong to a cluster spread their production to their neighbors
  for road in clusterRoads[-1]:
    neighbors = differentNeighbors(road)
    if len(neighbors) == 0:
      continue
    for good, amts in road.production.items():
      fractionAmts = [a / len(neighbors) for a in amts]
      for n in neighbors:
        n.production[good] = listAdd(n.production.get(good, []), fractionAmts)
  Timer.stop('Production')
  Timer.start('Cluster production')
  # Now each of the transportation roads distributes the resources of their cluster
  clusterResources = {}
  for cluster, roads in clusterRoads.items():
    if cluster < 0:
      continue
    clusterResources[cluster] = {}
    excessProduction, excessDemand = distributeResources(roads)
    clusterResources[cluster]['production'] = excessProduction
    clusterResources[cluster]['demand'] = excessDemand
  Timer.stop('Cluster production')
  # Now look at roads ceasing to be transport roads
  
  Timer.start('Road management')
  upgradeChoices = []
  downgradeChoices = []
  reskinChoices = []
  newclusterChoice = makeTransport(clusterRoads[-1], roadClusters)
  
  for cluster, roads in clusterRoads.items():
    if cluster > -1:
      transportRoads = list(filter(lambda r: type(r) is TransportRoad, roads))
      downgradeChoices.append(deTransport(transportRoads, roads, clusterResources[cluster]))
      upgradeChoices.append(reTransport(transportRoads, roads, clusterResources[cluster], clusterResources, roadClusters))
      reskinChoice, levelUpChoice = reskin(roads, clusterResources[cluster])
      reskinChoices.append(reskinChoice)
      Road.roadSet[levelUpChoice[0]].levelUp(levelUpChoice[1])
    
  Timer.stop('Road management')  
  Timer.start('Changing a road')
  # only do one of upgrade/downgrade/reskin
  if random.random() < len(clusterRoads[-1]) / len(roadClusters):
    choice = 4
  else:
    choice = random.random() * 3
  #choice = random.choice([0,2])
  
  
  if choice < 1: # upgrade
    luckyRoad, _ = sorted(upgradeChoices, key=lambda x: x[1])[-1]
    if luckyRoad is not None:
      luckyRoad = Road.roadSet[luckyRoad]
      logging.info('Upgrading road %s' % str(luckyRoad))
      convert(luckyRoad, TransportRoad)
  elif choice < 2: # downgrade
    neighbs = []
    i = 1
    while len(neighbs) == 0:
      if i > len(downgradeChoices):
        break
      luckyRoad, _ = sorted(downgradeChoices, key=lambda x: x[1])[-i]
      if luckyRoad is None:
        break
      luckyRoad = Road.roadSet[luckyRoad]
      neighbs = [r for r in chain(luckyRoad.start.roads, luckyRoad.end.roads) if type(r) is not TransportRoad and type(r) is not SpecialRoad]
      i += 1
    else:
      newType = type(random.choice(neighbs))
      logging.info('Downgrading road %s to %s' % (str(luckyRoad), str(newType)))
      convert(luckyRoad, newType)
  elif choice < 3: # reskin
    neighbs = []
    i= 1
    newtype = int
    luckyRoad = 5
    while newtype is type(luckyRoad):
      while len(neighbs) == 0:
        if i > len(reskinChoices):
          break
        luckyRoad, _ = sorted(reskinChoices, key=lambda x: x[1])[-i]
        if luckyRoad is None:
          break
        luckyRoad = Road.roadSet[luckyRoad]
        neighbs = [r for r in chain(luckyRoad.start.roads, luckyRoad.end.roads) if type(r) is not TransportRoad and type(r) is not SpecialRoad]
        i += 1
      if i > len(reskinChoices) or luckyRoad is None:
        break
      newType = type(random.choice(neighbs))
    else:
      logging.info('Changing road %s to %s' % (str(luckyRoad), str(newType)))
      convert(luckyRoad, newType)
  else: # new cluster
    luckyRoad, _ = newclusterChoice
    if luckyRoad is not None:
      luckyRoad = Road.roadSet[luckyRoad]
      logging.info('Making new cluster at %s'% str(luckyRoad))
      convert(luckyRoad, TransportRoad)

  Timer.stop('Changing a road')
  Timer.start('Resource counting')
  # Now to figure out what resource was lacking the most
  resources = {}
  for cluster, output in clusterResources.items():
    for good, amts in output['production'].items():
      resources[good] = listAdd(resources.get(good, []), amts)
    for good, amts in output['demand'].items():
      resources[good] = listMinus(resources.get(good,[]), amts)
  
  for road in clusterRoads[-1]:
    for good, amts in road.production.items():
      resources[good] = listAdd(resources.get(good, []), amts)
    for good, amts in road.baseDemand.items():
      resources[good] = listMinus(resources.get(good, []), amts)
  
  
  
  bestGood, amt = sorted(resources.items(), key=lambda x: abs(x[1][0]))[-1]
  amt = amt[0]
  Timer.stop('Resource counting')
  Timer.start('Make a new road')
  logging.info('Remaining goods: %s, %f' % (str(bestGood), amt))
  if abs(amt) > 0:
    if amt < 0:
      roadType = GoodType.roadSupply(bestGood)
    else:
      roadType = GoodType.roadDemand(bestGood)
    
    if random.random() > 0.5:
      createNewRoad(roadType, abs(amt))
    elif random.random() > 0.01:
      createNewRoadFillIn(roadType, abs(amt))
    else:
      useful = list(filter(lambda x: type(x) is roadType, Road.roadSet.values()))
      if len(useful) > 0:
        luckyRoad = random.choice(useful)
        luckyRoad.level += 1
      else:
        createNewRoad(roadType, abs(amt))
  else:
    luckyRoad = random.choice(list(Road.roadSet.values()))
    while type(luckyRoad) is SpecialRoad:
      luckyRoad = random.choice(list(Road.roadSet.values()))
    luckyRoad.level += 1
  Timer.stop('Make a new road')
  Timer.start('Ending cleanup')
  cleanNodes(Node.nodeSet)
  cleanRoads(Road.roadSet)
  assert Node.tree.sanityCheck()
  Timer.stop('Ending cleanup')

def findClosePair(nodeSet, roadSet): # find the node and the (node or edge) that are clsoest together
  logging.info('Finding closest pair of nodes')
  closePair = (None, None)
  closeRelativeDistance = np.infty
  closeDirectDistance = np.infty
  
  closePair1, closeRelativeDistance1, closeDirectDistance1 = closePair, closeRelativeDistance, closeDirectDistance
  
  # Only need to look at nodes that were recently added
  Timer.start('Recently added nodes')
  for node1 in Node.getRecents():
    closePair1, closeRelativeDistance1, closeDirectDistance1 = closeNode(node1, nodeSet, roadSet)
    if closeRelativeDistance1 < closeRelativeDistance or (closeRelativeDistance1 == closeRelativeDistance and closeDirectDistance1 < closeDirectDistance):
      logging.info('Master (Node): Found a good pair: %s and %s' % (str(closePair1[0]), str(closePair1[1])))
      closeRelativeDistance = closeRelativeDistance1
      closeDirectDistance = closeDirectDistance1
      closePair = closePair1
  Timer.stop('Recently added nodes')
  
  Timer.start('Recently added roads')
  for road1 in Road.getRecents():
    closePair1, closeRelativeDistance1, closeDirectDistance1 = closeRoad(road1, nodeSet, roadSet)
    if closeRelativeDistance1 < closeRelativeDistance or (closeRelativeDistance1 == closeRelativeDistance and closeDirectDistance1 < closeDirectDistance):
      logging.info('Master (Road): Found a good pair: %s and %s' % (str(closePair1[0]), str(closePair1[1])))
      closeRelativeDistance = closeRelativeDistance1
      closeDirectDistance = closeDirectDistance1
      closePair = closePair1
  Timer.stop('Recently added roads')
  


  logging.info('Closest pair is %s to %s, with a relative distance of %5f and direct distance of %5f' % (str(closePair[0]), str(closePair[1]), closeRelativeDistance, closeDirectDistance))  
  return closePair, closeRelativeDistance
  
  
def closeNode(node1, nodeSet, roadSet):
  id1 = node1.id
  distances, prev = dijkstra(id1, nodeSet)
  logging.info(str(distances))
  closePair = (None, None)
  closeRelativeDistance = np.infty
  closeDirectDistance = np.infty
  Timer.start('Node to node')
  # Gather the nearby nodes
  nearbyNodes = [(n.id, n) for n in Node.tree.getNode(node1).getPointsFromNeighbors()]
  logging.info("Looking to connect %s with any of %s" % (str(node1), ', '.join(str(n) for _,n in nearbyNodes)))
  for id2, node2 in nearbyNodes: # find the closest node
    if id2 == id1:
      continue
    if np.isclose(dist(node1, node2), 0) or min([abs(angle(node1, node2) - r.getAngle()) for r in node2.roads], default=np.infty) < 0.7:
      #logging.info('%s is too collinear with a road coming out of %s' % (node1, node2))
      continue
    directDistance = dist(node1, node2)
    relativeDistance = directDistance / distances[node2.id]
    if relativeDistance +1e-9 < closeRelativeDistance:
      logging.info('Node: Found a close pair: %s is close to %s' % (str(node1), str(node2)))
      closeRelativeDistance = relativeDistance
      closeDirectDistance = directDistance
      closePair = (node1, node2)
    elif relativeDistance == closeRelativeDistance and directDistance + 1e-9< closeDirectDistance:
      logging.info('Node: Found a more direct close pair: %s is close to %s' % (str(node1), str(node2)))
      closeRelativeDistance = relativeDistance
      closeDirectDistance = directDistance
      closePair = (node1, node2)
  Timer.stop('Node to node')
  Timer.start('Node to road')
  nearbyRoads = [(r.id, r) for r in Node.tree.getNode(node1).getRoads(1)]
  for id2, road in nearbyRoads:
    pt1, pt2 = road.endpoints
    
    if road in node1.roads or np.isclose(dist(node1.coord, pt1),0) or np.isclose(dist(node1.coord, pt2),0):
      continue
    logging.info('Road %s not connected to node %s: %s' % (str(road), str(node1), ', '.join(str(r) for r in node1.roads)))
    if angle(pt1, pt2, node1.coord)*2 < np.pi and angle(pt2, pt1, node1.coord)*2 < np.pi: # we're in between, need to find point of intersection
      inter, directDistance, closeEndpoint = findIntersection(road, node1)
      normalDistance = distances[closeEndpoint.id] + dist(closeEndpoint, inter) * road.travelTime() / road.length
      relativeDistance = directDistance / normalDistance
      if abs(angle(node1, inter) - road.getAngle()) < 0.1:
        continue
      elif relativeDistance + 1e-9 < closeRelativeDistance:
        logging.info('Node: Found a close pair: %s is close to road %s at %s' % (str(node1), str(road), str(inter)))
        closeRelativeDistance = relativeDistance
        closeDirectDistance = directDistance
        closePair = (node1, inter)
      elif relativeDistance == closeRelativeDistance and directDistance + 1e-9 < closeDirectDistance:
        logging.info('Node: Found a more direct close pair: %s is close to road %s at %s' % (str(node1), str(road), str(inter)))
        closeRelativeDistance = relativeDistance
        closeDirectDistance = directDistance
        closePair = (node1, inter)
  Timer.stop('Node to road')  
  return closePair, closeRelativeDistance, closeDirectDistance
  
def closeRoad(road, nodeSet, roadSet):
  id2 = road.id
  # pt1, pt2 = road.endpoints
  pt1, pt2 = road.start, road.end

  closePair = (None, None)
  closeRelativeDistance = np.infty
  closeDirectDistance = np.infty
  Timer.start('Road to node')
  Timer.start('Collecting adjacent nodes')
  nearbyNodes = [(n.id, n) for t in Road.tree.getRoadNodes(road) for n in t.getPointsFromNeighbors() ]
  seen = set()
  Timer.stop('Collecting adjacent nodes')
  for id1, node1 in nearbyNodes: # find the closest node
    if id1 in seen:
      continue
    seen.add(id1)
    distances, prev = dijkstra(id1, nodeSet)
    logging.info(str(distances))

    
    if road in node1.roads:
      continue
    logging.info('Road: Road %s not connected to node %s: %s' % (str(road), str(node1), ', '.join(str(r) for r in node1.roads)))
    if angle(pt1, pt2, node1)*2 < np.pi and angle(pt2, pt1, node1)*2 < np.pi: # we're in between, need to find point of intersection
      inter, directDistance, closeEndpoint = findIntersection(road, node1)
      normalDistance = distances[closeEndpoint.id] + dist(closeEndpoint, inter) * road.travelTime() / road.length
      relativeDistance = directDistance / normalDistance
      if abs(angle(node1, inter) - road.getAngle()) < 0.1:
        continue
      elif relativeDistance + 1e-9 < closeRelativeDistance:
        logging.info('Road: Found a close pair: %s is close to road %s at %s' % (str(node1), str(road), str(inter)))
        closeRelativeDistance = relativeDistance
        closeDirectDistance = directDistance
        closePair = (node1, inter)
      elif relativeDistance == closeRelativeDistance and directDistance + 1e-9 < closeDirectDistance:
        logging.info('Road: Found a more direct close pair: %s is close to road %s at %s' % (str(node1), str(road), str(inter)))
        closeRelativeDistance = relativeDistance
        closeDirectDistance = directDistance
        closePair = (node1, inter)
  Timer.stop('Road to node')  
  return closePair, closeRelativeDistance, closeDirectDistance
  
def dijkstra(source, nodeSet):
  distances = {i: np.infty for i in nodeSet}
  prev = {i: -1 for i in nodeSet}
  Q = set(nodeSet.keys())
  
  distances[source] = 0
  while len(Q) > 0:
    u = sorted(Q, key=lambda x: distances[x])[0]
    Q.remove(u)
    for road in nodeSet[u].roads:
      neighb = road.otherEndpoint(u)
      alt = distances[u] + road.travelTime()
      if alt < distances[neighb]:
        distances[neighb] = alt
        prev[neighb] = u
  
  return distances,prev

def dijkstraMultiplier(source, roadSet):
  multipliers = {i: 0 for i in roadSet}
  prev = {i: -1 for i in roadSet}
  Q = set(roadSet.keys())
  
  multipliers[source.id] = 1
  while len(Q) > 0:
    u = sorted(Q, key=lambda x: multipliers[x])[-1]
    assert multipliers[u] > 0, (u, source.id)
    Q.remove(u)
    for road in chain(roadSet[u].start.roads, roadSet[u].end.roads):
      alt = multipliers[u] * road.transportMultiplier()
      assert alt > 0, (source.id, str(u), multipliers[u], str(road), road.transportMultiplier()) 
      if alt > multipliers[road.id]:
        multipliers[road.id] = alt
        prev[road.id] = u
  
  
  for k in multipliers:
    multipliers[k] /= roadSet[k].transportMultiplier()
  multipliers[source.id] = 1
  return {'mult': multipliers, 'prev': prev}
  
  
  
def findIntersection(road, pt):
  pt1, pt2 = road.start, road.end
  assert dist(pt1,pt) * dist(pt2, pt) > 0, (str(road), str(pt), road in pt.roads)
  angle1 = angle(pt2, pt1, pt)
  angle2 = angle(pt1, pt2, pt)
  if closeGreater(angle1,0) and closeGreater(np.pi, angle1*2) and closeGreater(angle2,0) and closeGreater(np.pi, angle2*2): # find point of intersection
    grad = pt2.coord - pt1.coord
    perp = np.array([-grad[1], grad[0]])
    t1, t2 = -np.linalg.solve( np.vstack([grad, perp]).T, pt1.coord - pt.coord)
    assert t1 >= 0, (t1, str(pt1), str(pt2), str(pt))
    assert t1 <= 1, (t1, str(pt1), str(pt2), str(pt))
    assert closeGreater(abs(t2),0), (t1, str(pt1), str(pt2), str(pt))
    if t1 * road.length < Road.minLength:
      logging.info('Too close to start node')
      return pt1, dist(pt1, pt), pt1
    elif (1-t1) * road.length < Road.minLength:
      logging.info('Too close to end node')
      return pt2, dist(pt2, pt), pt2
    inter = Node(pt1.coord + t1*grad, road=road)
    if min( [abs(angle(pt,inter) - r.getAngle()) for r in pt.roads], default=1) < 0.5:
      logging.info('Not worth building a road for this')
      return inter, np.infty, pt1
    closeEndpoint = pt1
    if t1 <  0.5:
      closeEndpoint = pt2
    return inter, dist(inter, pt), closeEndpoint
  
  else:
    d1 = dist(pt1, pt)
    d2 = dist(pt2, pt)
    if d1 < d2:
      return pt1, d1, pt1
    else:
      return pt2, d2, pt2

def roadIntersection(end1, end2, road):
  logging.info('Want road %s to %s, checking intersection with %s' % (str(end1), str(end2), str(road)))
  grad1 = end2.coord - end1.coord
  
  pt1, pt2 = road.start, road.end
  grad2 = pt2.coord - pt1.coord
  
  gradients = np.vstack([grad1, -grad2]).T
  
  if np.linalg.det(gradients) == 0:
    if dist(end2, road.start) < 5:
      return road.start
    elif dist(end2, road.end) < 5:
      return road.end
    else:
      return end2
  t1, t2 = np.linalg.solve(gradients, pt1.coord - end1.coord)

  if t1 < 1 and t1 > 0 and t2 <= 1 and t2 >= 0:
    if t2 * road.length < 5:
      return road.start
    elif (1-t2) * road.length < 5:
      return road.end
    inter = Node(pt1.coord + t2*grad2, road=road) 
    logging.info('Intersection t=(%f,%f) with %s found at %s' % (t1, t2, str(road), str(inter)))
    return inter
  else:
    return end2
    
def makeRoad(newStart, newEnd, roadSet, roadType): # Adjust newEnd if necessary to avoid intersections
  logging.info('Attempting to make road from %s to %s' % (str(newStart), str(newEnd)))
  for road in roadSet.values():
    newEnd = roadIntersection(newStart, newEnd, road)
  
  if not newEnd.isActive():
    newEnd.add()
  assert abs(newStart.coord - newEnd.coord).sum() > 0, (str(newStart), str(newEnd))
  road = roadType(newStart, newEnd)
  road.add()

  return road
  
def drawCity(fname, clean=True, clusters=True):

  logging.info('Saving %s' % fname)
  img = Image.new('RGBA', (400,400), (255,255,255,255))
  draw = ImageDraw.Draw(img)
  
  xMin = min(n.coord[0] for n in Node.nodeSet.values()) - 50
  yMin = min(n.coord[1] for n in Node.nodeSet.values()) - 50
  xMax = max(n.coord[0] for n in Node.nodeSet.values()) + 50
  yMax = max(n.coord[1] for n in Node.nodeSet.values()) + 50

  upperLeft = np.array([xMin, yMin])
  scale = min(1, 400/(xMax - xMin), 400/(yMax - yMin))
  
  if clusters:
    clusterRoads, roadClusters = assignClusters(Node.nodeSet, Road.roadSet)
    clusterColors = random.choices([c for c in ImageColor.colormap if c != 'black' and min(ImageColor.getrgb(c)) < 170], k=len(clusterRoads)-1)    
    clusterColors.append('black')
  
  logging.info('Roads:')
  for road in Road.roadSet.values():
    logging.info(str(road))
    if clusters:
      color = ImageColor.colormap[clusterColors[roadClusters[road.id]]]
      if type(road) is TransportRoad:
        #color = (min(10, color[0]-20), max(10, color[1]-20), max(10, color[2]-20))
        pass
      draw.line([tuple(scale*(road.start.coord-upperLeft)), tuple(scale*(road.end.coord-upperLeft))], fill=color, width=int(np.sqrt(2*road.level)))
    else:
      draw.line([tuple(scale*(road.start.coord-upperLeft)), tuple(scale*(road.end.coord-upperLeft))], fill=road.color(), width=int(np.sqrt(2*road.level)))
  
  
  logging.info('\nNodes:')
  for n in Node.nodeSet.values():
    logging.info(str(n))
    x = scale*(n.coord[0] - upperLeft[0])
    y = scale*(n.coord[1] - upperLeft[1])
    draw.ellipse([(x-1,y-1), (x+1,y+1)], fill="black")
    if not clean:
      draw.text((x,y-10), str(n.id), fill='black')
  img.save(fname)


def split(roadSet):
  longestRoad = None
  longest = 0
  for road in roadSet.values():
    if road.length > longest:
      longest = road.length
      longestRoad = road
  
  if longest > 100:
    logging.info('Longest road %s has angle %f' % (str(longestRoad), longestRoad.getAngle()))
    fraction = (random.random()+1)/3
    breakPoint = longestRoad.start.coord + fraction*(longestRoad.end.coord - longestRoad.start.coord)
    breakNode = Node(breakPoint[0], breakPoint[1], road=longestRoad)
    logging.info('Breaking %s at %s' % (str(longestRoad), str(breakNode)))
    breakNode.add()
    
  
def cleanNodes(nodeSet):  
  # Second, merge close nodes
  nodeList = list(n for n in nodeSet)
  i = 0
  while i < len(nodeList):
    node1 = nodeSet[nodeList[i]]
    for id, node2 in nodeSet.items():
      if id == node1.id:
        continue
      assert id > 0, str(node)
      assert node2.id > 0, str(node)
      assert id == node2.id, str(node)
      if dist(node1, node2) < Road.minLength:
        # need to combine
        if len(node1.roads) > len(node2.roads):
          nodeList.remove(node2.id)
          node2.replace(node1)
        else:
          nodeList.remove(node1.id)
          node1.replace(node2)
        assert len(nodeList) == len(nodeSet), (len(nodeList), len(nodeSet), id, node1.id, nodeList, nodeSet.keys())
        break
    else:
      i += 1

def cleanRoads(roadSet):
  # Don't want any too-small angles
  roadList = list(r for r in roadSet)
  i = 0
  while i < len(roadList):
    road1 = roadSet[roadList[i]]
    toremove = None
    for road2 in road1.start.roads:
      if road2.id == road1.id:
        continue
      a = angle(road1.end, road1.start, road2.otherEndpoint(road1.start)) 
      if  a < np.pi / 12: # gotta remove one of these roads
        logging.info('Angle %.3f is too small between %s and %s, removing one' % (a, str(road1), str(road2)))
        toremove = random.choice([road1, road2])
        break
    else:
      for road2 in road1.end.roads:
        if road2.id == road1.id:
          continue
        a = angle(road1.start, road1.end, road2.otherEndpoint(road1.end))
        if a < np.pi / 12: # gotta remove one of these roads
          logging.info('Angle %.3f is too small between %s and %s, removing one' % (a, str(road1), str(road2)))
          toremove = random.choice([road1, road2])
          break
    if toremove is not None:
      roadList.remove(toremove.id)
      toremove.remove()
    else:
      i += 1

def createNewRoadFillIn(roadType, amt):
  # New idea (uncomment double comment above to revert)
  # Choose n random pairs of points, and pick the pair for which the midpoint has the smallest density of points (maybe parent has smallest density)
  
  closeDirectDistance = 0
  attempts = 0
  while closeDirectDistance < max(amt, Road.minLength) and attempts < 10:
    attempts += 1
    bestDensity = np.inf
    newStart = None
    for _ in range(20):
      aRoads = set([0])
      bRoads = set([0])
      attempts2 = 0
      while len(aRoads.intersection(bRoads)) > 0 and attempts2 < 10:
        attempts2 += 0
        a, b = random.sample(Node.nodeSet.keys(), 2)
        aNode = Node.nodeSet[a]
        bNode = Node.nodeSet[b]
        aRoads = set([r.id for r in aNode.roads])
        bRoads = set([r.id for r in bNode.roads])
      if attempts == 0:
        continue
      proposed = Node((aNode.coord + bNode.coord)/2)
      treeNode = Node.tree.getNode(proposed).parent
      density = treeNode.pointCount / (treeNode.maxX - treeNode.minX)**2
      if density < bestDensity:
        bestDensity = density
        newStart = proposed
    
    # Now I need to figure out where to connect it to
    (node1, node2), closeRelativeDistance, closeDirectDistance = closeNode(proposed, Node.nodeSet, Road.roadSet)
    if np.isclose(0, np.square(node2.coord - proposed.coord).sum()): 
      node1, node2 = node2, node1
  
  if closeDirectDistance >= max(amt, Road.minLength):
    node1.add()
    makeRoad(node1, node2, Road.roadSet, roadType)
        
      
def createNewRoad(roadType, amt):  
# randomly choose a node with probability proportional to how sparse it is
  newLevel = 1
  amt = min(amt, 40)
  logging.info('Creating new %s of length %.3f' % (str(roadType), amt))
  
  # probs = {}
  # for node in Node.nodeSet.values():
    # if len(node.roads) < 5 and node.unlocked:
      # probs[node.id] = 1. / len(node.roads)

  # assert len(probs) > 0, '\n'.join([': '.join((str(n), str(len(n.roads)), '\n'.join(map(str, n.roads)))) for n in Node.nodeSet.values()])
  # choice = random.random() * sum(v for v in probs.values())
  # assert choice > 0, probs
  # luckyNode = -1
  # while choice > 0:
    # luckyNode, weight = probs.popitem()
    # choice -= weight
  
  treeNode = Node.tree.root
  nodeMatrix = np.array([n.coord for n in Node.nodeSet.values()])
  n = nodeMatrix.shape[0]
  center = np.dot(np.ones((n,1)), np.dot(np.ones((1,n))/n, nodeMatrix))
  centeredNodes = nodeMatrix - center
  eigs, evecs = np.linalg.eig( np.dot(centeredNodes.T, centeredNodes))

  eigendata = sorted(zip(eigs,evecs), key=lambda x: x[0])
  eigs = [v for v,_ in eigendata]
  evecs = [w for _,w in eigendata]

  
  while not treeNode.isLeaf:
    probs = [ abs(np.dot(evecs[0], np.array([child.middleX, child.middleY])) * eigs[1] /(max(len(child.children),1) *eigs[0] * np.sqrt(child.pointCount))) if child.pointCount > 0 else 0 for child in treeNode.children]

    treeNode = np.random.choice(treeNode.children, p=[p/sum(probs) for p in probs])
  

  
  newStart = random.choice(treeNode.data)
  
  
  #newStart = node.nodeSet[luckyNode]
  logging.info('Putting road on %s' % (str(newStart)))
  # choose a direction that is pretty far from the other nodes coming out
  logging.info('Other roads are: %s' % ', '.join(map(str,newStart.roads)))
  angles = sorted(enumerate([r.getAngle(newStart) for r in newStart.roads]), key=lambda x: x[1])
  
  assert len(angles) > 0, str(newStart)
  angles += [(angles[0][0], angles[0][1] + 2*np.pi)] # to complete the cycle
  logging.info('Angles: %s' % str(angles))
  intervals = [(angles[i-1][0], angles[i][1] - angles[i-1][1]) for i in range(1,len(angles))] # the first entry tells you what road starts the interval
  if len(intervals) == 0:
    intervals = [(0, 2*np.pi)]
    
  assert abs(sum(i[1] for i in intervals) - 2*np.pi) < 1e-5, intervals
  logging.info('Intervals: %s' % str(intervals))
  intervalIndex, (angleIndex1, bigInterval) = sorted(enumerate(intervals), key=lambda x: x[1][1])[-1]
  angleIndex2 = intervals[(intervalIndex+1) % len(intervals)][0]
  
  logging.info('Largest gap is between %f and %f, for a size of %f' % (angles[intervalIndex][1], angles[(intervalIndex+1)][1], bigInterval))
  logging.info('That corresponds to %s and %s' % (str(newStart.roads[angleIndex1]), str(newStart.roads[angleIndex2])))
  # if it's a spoke, I don't want to continue in that direction
  if len(angles) == 2: # between 1/6 and 2/6 or 4/6 and 5/6
    fraction = (random.choice([1,4]) + random.random())/6
  else:
    fraction = (1 + random.random())/3
  logging.info('Going %3f of the way' % fraction)
  roadAngle = (bigInterval * fraction + angles[intervalIndex][1]) % (2*np.pi)
  
  logging.info('Chose angle %f' % roadAngle)
  newEnd = Node((np.array([np.cos(roadAngle), np.sin(roadAngle)]) * abs(amt)) + newStart.coord)
  
  logging.info('Placing %s' % str(newEnd))
  
  
  if len(angles) == 2:
    #assert roadAngle - angles[0] > 2*np.pi / 3, (roadAngle, angles)
    assert angle(newEnd, newStart, newStart.roads[0].otherEndpoint(newStart)) + 1e-5 > np.pi / 3, (str(newEnd), str(newStart), str(newStart.roads[0]), angle(newEnd, newStart, newStart.roads[0].otherEndpoint(newStart)))
    assert angle(newEnd, newStart, newStart.roads[0].otherEndpoint(newStart)) + 1e-5 <  2*np.pi / 3, (str(newEnd), str(newStart), str(newStart.roads[0]), angle(newEnd, newStart, newStart.roads[0].otherEndpoint(newStart)))
    pass
  else:
    angle1 = angle(newEnd, newStart, newStart.roads[angleIndex1].otherEndpoint(newStart))
    angle2 = angle(newEnd, newStart, newStart.roads[angleIndex2].otherEndpoint(newStart))
    assert 2*angle1 + 1e-5 > angle2, (angle1, angle2, str(newStart.roads[angleIndex1].otherEndpoint(newStart)), str(newStart.roads[angleIndex2].otherEndpoint(newStart)))
    assert 2*angle2 + 1e-5 > angle1, (angle1, angle2, str(newStart.roads[angleIndex1].otherEndpoint(newStart)), str(newStart.roads[angleIndex2].otherEndpoint(newStart)))
  newRoad = makeRoad(newStart, newEnd, Road.roadSet, roadType)
  newRoad.level = newLevel    

def assignClusters(nodeSet, roadSet):
  # Now assign clusters
  # First, connect transport roads
  clusterRoads, roadClusters = collectTransportRoads(roadSet)
  clusterRoads[-1] = []
  # Now assign every road to a transport road
  for road in roadSet.values():
    if road.id not in roadClusters:
      closestTransport = findTransport(road)
      if closestTransport is None:
        clusterRoads[-1].append(road)
        roadClusters[road.id] = -1
      else:
        cluster = roadClusters[closestTransport.id]
        clusterRoads[cluster].append(road)
        roadClusters[road.id] = cluster
  return clusterRoads, roadClusters
        
def collectTransportRoads(roadSet):
  clusterRoads = {}
  roadClusters = {}
  
  for road in roadSet.values():
    if road.id not in roadClusters and type(road) is TransportRoad:
      cluster = collectNeighbors(road, {road.id: road})
      for r in cluster:
        roadClusters[r] = len(clusterRoads)
      clusterRoads[len(clusterRoads)] = list(cluster.values())
  
  #return clusters, clusterAssignment
  return clusterRoads, roadClusters
  
def collectNeighbors(road, assigned):
  for neighb in chain(road.start.roads, road.end.roads):
    if neighb.id not in assigned and type(road) is TransportRoad:
      assigned[neighb.id] = neighb
      assigned = collectNeighbors(neighb, assigned)
  return assigned
  
def findTransport(road):
  roads = [r for r in chain(road.start.roads, road.end.roads)]
  visited = {r.id for r in roads}
  secondLayer = []
  for r in roads:
    if type(r) is TransportRoad:
      return r
    for r2 in chain(r.start.roads, r.end.roads):
      if r2.id not in visited:
        secondLayer.append(r2)
  for r in secondLayer:
    if type(r) is TransportRoad:
      return r
  
  return None

def differentNeighbors(road):
  neighbors = []
  roads = [r for  r in chain(road.start.roads, road.end.roads)]
  visited = {r.id for r in roads}
  i = 0
  while i < len(roads):
    r = roads[i]
    if type(r) == type(road):
      for r2 in chain(r.start.roads, r.end.roads):
        if r2.id not in visited:
          roads.append(r2)
          visited.add(r2.id)
    else:
      neighbors.append(r)
    i += 1
  return neighbors
  
  
def distributeResources(roads):
  resources = {}
  
  for road in roads:
    for good, amts in road.production.items():
      resources[good] = listAdd(resources.get(good, []), amts)
  
  roads = sorted(roads, key=lambda r: r.length)
  roads = sorted(roads, key=lambda r: -r.level)
  
  for road in roads:
    for good, amts in road.supplies.items():
      if good not in resources:
        continue
      for level, amt in enumerate(amts):
        if len(resources[good]) > level:
          request = min(amt, resources[good][level])
          road.supplies[good][level] -= request
          resources[good][level] -= request
  
  demand = {}
  for road in roads:
    for good, amts in road.supplies.items():
      demand[good] = listAdd(demand.get(good,[]), amts)
  
  
  
  return resources, demand  

  
def findLoops(roads):
  loops = []
  # Need to create a sort of degenerate node
  nodes = {}
  source = None
  for r in roads:
    if r.start.id not in nodes:
      nodes[r.start.id] = {'roads': [r], 'node': r.start}
    else:
      nodes[r.start.id]['roads'].append(r)
      
    if r.end.id not in nodes:
      nodes[r.end.id] = {'roads': [r], 'node': r.end}
    else:
      nodes[r.end.id]['roads'].append(r)
      
  source = roads[0].start.id
  distances = {n: np.infty for n in nodes}
  prev = {n: -1 for n in nodes}
  Q = set(nodes)
  
  distances[source] = 0
  while len(Q) > 0:
    u = sorted(Q, key=lambda x: distances[x])[0]
    Q.remove(u)
        
    for road in nodes[u]['roads']:
      neighb = road.otherEndpoint(nodes[u]['node'])
      alt = distances[u] + 1
      if distances[neighb.id] < np.infty and distances[neighb.id] + 2 < alt: # have seen it before and is actually a loop
        pU = [u]
        backtrack = prev[u]
        while backtrack > -1:
          pU.append(backtrack)
          backtrack = prev[backtrack]
        backtrack = prev[neighb.id]
        pN = [neighb.id]
        while backtrack > -1:
          pN.append(backtrack)
          backtrack = prev[backtrack]
        loops.append(joinLoop(pU, pN))
      elif alt < distances[neighb.id]: # haven't seen it before
        distances[neighb.id] = alt
  return loops
      
def joinLoop(a,b):
  intercept = set(a)
  
  for n in b:
    if n in intercept:
      meet = n
      break
  else:
    raise Exception('no intersection')
  
  loop = set([meet])
  for path in [a,b]:
    for n in path:
      if n == meet:
        break
      loop.add(n)
  
  return loop

def marginalSupply(transportRoads, allRoads, resources):
  roadCounts = {}
  roadNeighbors = {}
  
  for road in transportRoads: # each road contributes the roads connected to it and all the roads connected to those
    neighbs = [r for r in chain(road.start.roads, road.end.roads) if r.id in allRoads]
    visited = {r.id: r for r in neighbs}
    for r in neighbs:
      for r2 in chain(r.start.roads, r.end.roads):
        if r2.id not in visited and r2.id in allRoads:
          visited[r2.id] = r2
    roadNeighbors[road.id] = visited
    for r, road in visited:
      roadCounts[r] = roadCounts.get(r, 0) + 1
      
  
  roadDemands = {}
  roadSupplies = {}
  for road in transportRoads:
    demands = {}
    supplies = {}
    for r, neighb in roadNeighbors[road.id].items():
      if roadCounts[r] == 1 and type(neighb) is not TransportRoad:
        for good, amts in neighb.baseProduction.items():
          supplies[good] = listAdd(supplies.get(good,[]), amts)
        for good, amts in neighb.baseDemand.items():
          demands[good] = listAdd(demands.get(good, []), amts)
    roadDemands[road.id] = demands
    roadSupplies[road.id] = supplies
    
      
      
  # Now that I know what resources are unique to what road,
  # I can compute how much production and demand it wastes
  # What do I want to compute?
  # The more unused supply there is, the more I should penalize supplying that
  # So I'm going to compute supply * unusedSupply
  # But I should also compute supply * unmetDemand and reward this
  # Subtract them, so positive is bad
  # I would like to normalize these numbers, how about by whatever the excess supply/demand is for that good?
  
  # normalizer = copy.deepcopy(resources['demand'])
  # for good, amts in resources['production'].items():
    # normalizer[good] = listAdd(normalizer.get(good, []), amts)
  
  roadWasteSupply = {}
  for rID, supplies in roadSupplies.items():
    supply = {}
    for good, amts in supplies.items():
      badScore = listMultiply(amts, resources['production'].get(good, []))
      goodScore = listMultiply(amts, resources['demand'].get(good, []))
      
      supply[good] = listMinus(badScore, goodScore)
    roadWasteSupply[rID] = supply
  roadWasteDemand = {}        
  for rID, demands in roadDemands.items():
    demand = {}
    for good, amts in demands.items():
      badScore = listMultiply(amts, resources['demand'].get(good, []))
      goodScore = listMultiply(amts, resources['production'].get(good, []))
      demand[good] = listMinus(badScore, goodScore)
    roadWasteDemand[rID] = demand
      
  return roadWasteDemand, roadWasteSupply
  
def deTransport(transportRoads, allRoads, resources):
  loops = findLoops(transportRoads) # transport roads that are part of a loop will want to die
  # also roads that don't supply much
  # also give a multiplier based on how spread out the cluster is
  
  randomNodes = set([r.start for r in random.choices(allRoads, k=10)])
  distances = [dist(a,b) for a in randomNodes for b in randomNodes if b.id > a.id]
  if len(distances) > 1:
    mult = max(distances) * len(distances) / sum(distances)
  else:
    mult = 1
  
  
  wasteDemand, wasteSupply = marginalSupply(transportRoads, allRoads, resources)
  
  scores = {}
  for road in transportRoads:
    loopCount = sum( road.id in l for l in loops)
    resourceCount = sum(sum(amts) for amts in chain(wasteDemand[road.id].values(),wasteSupply[road.id].values()))
    scores[road.id] = loopCount + resourceCount
    # but also subtract off the number of transportRoads it is connected to 
    neighborRoads = sum( type(r) is TransportRoad for r in chain(road.start.roads, road.end.roads))
    scores[road.id] -= neighborRoads
    scores[road.id] *= mult
  return sorted(scores.items(), key=lambda x: x[1])[-1] # return the transport road that should be demoted
  
def reTransport(transportRoads, allRoads, resources, allResources, roadClusters):
  # A road should become a transport road if
  # 0. It is connected to a transport road
  # 1. It connects needed roads
  # 2. It connects to another cluster that provides needed resources
  # 3. It does not introduce a loop
  cluster = roadClusters[transportRoads[0].id]
  candidates = []
  loopMakers = set()
  seen = set()
  for road in transportRoads:
    
    for neighb in chain(road.start.roads, road.end.roads):  
      if type(neighb) is not TransportRoad and type(neighb) is not SpecialRoad and neighb.id not in seen:
        candidates.append(neighb)
        seen.add(neighb.id)
        if type(neighb) is TransportRoad and roadClusters[neighb.id] == cluster:
          loopMakers.add(neighb.id)
        
  if len(candidates) == 0:
    return (None, -np.inf)
  wasteDemand, wasteSupply = marginalClusterSupply(candidates, transportRoads, allRoads, resources, allResources, roadClusters)
  scores = {}
  for road in candidates:
    resourceCount = sum(sum(amts) for amts in chain(wasteDemand[road.id].values(), wasteSupply[road.id].values()))
    scores[road.id] = resourceCount
    if road.id in loopMakers:
      scores[road.id] += 100
  
  return sorted(scores.items(), key=lambda x: x[1])[-1] # return the candidate road that should be made into a transport road
  
def marginalClusterSupply(candidates, transportRoads, allRoads, resources, allResources, roadClusters):
  # Determining what resources a road brings in is actually kind of easy; count all the roads within 2 links that are not already part of a cluster
  # First, what cluster are we in?
  cluster = roadClusters[transportRoads[0].id]
  
  roadNeighbors = {}
  roadConnections = {}
  
  for road in candidates: # each road contributes the roads connected to it and all the roads connected to those
    neighbs = [r for r in chain(road.start.roads, road.end.roads)]
    visited = {r.id: r for r in neighbs}
    connections = set([roadClusters[r.id] for r in neighbs if type(r) is TransportRoad])
    for r in neighbs:
      for r2 in chain(r.start.roads, r.end.roads):
        if r2.id not in visited and r2.id in allRoads:
          visited[r2.id] = r2
    roadNeighbors[road.id] = {k: v for k,v in visited.items() if roadClusters[k] not in connections}
    connections.remove(cluster)
    roadConnections[road.id] = connections
    
    
  
  roadDemands = {}
  roadSupplies = {}
  for road in candidates:
    demands = {}
    supplies = {}
    
    for r, neighb in roadNeighbors[road.id].items():
      for good, amts in neighb.baseProduction.items():
        supplies[good] = listAdd(supplies.get(good,[]), amts)
      for good, amts in neighb.baseDemand.items():
        demands[good] = listAdd(demands.get(good, []), amts)
    for c in roadConnections[road.id]:
      for good, amts in allResources[c]['demand'].items():
        demands[good] = listAdd(demands.get(good, []), amts)
      for good, amts in allResources[c]['production'].items():
        supplies[good] = listAdd(supplies.get(good, []), amts)
        
    roadDemands[road.id] = demands
    roadSupplies[road.id] = supplies
    
      
      
  # Now that I know what resources are unique to what road,
  # I can compute how much production and demand it wastes
  # What do I want to compute?
  # The more unused supply there is, the more I should penalize supplying that
  # So I'm going to compute supply * unusedSupply
  # But I should also compute supply * unmetDemand and reward this
  # Subtract them, so positive is bad
  # I would like to normalize these numbers, how about by whatever the excess supply/demand is for that good?
  
  
  roadWasteSupply = {}
  for rID, supplies in roadSupplies.items():
    supply = {}
    for good, amts in supplies.items():
      badScore = listMultiply(amts, resources['production'].get(good, []))
      goodScore = listMultiply(amts, resources['demand'].get(good, []))
      
      supply[good] = listMinus(badScore, goodScore)
    roadWasteSupply[rID] = supply
  roadWasteDemand = {}        
  for rID, demands in roadDemands.items():
    demand = {}
    for good, amts in demands.items():
      # demand[good] = listDivide(listMinus(resources['production'].get(good, []), amts), normalizer[good])
      badScore = listMultiply(amts, resources['demand'].get(good, []))
      goodScore = listMultiply(amts, resources['production'].get(good, []))
      demand[good] = listMinus(badScore, goodScore)
    roadWasteDemand[rID] = demand
      
  return roadWasteDemand, roadWasteSupply
  
  
def makeTransport(roads, roadClusters):
  # A road that is not part of a cluster will become a transport road if
  # 1. A supermajority of its neighbors are not part of a cluster
  # 2. Its neighbors are diverse in type
  
  if len(roads) == 0:
    return (None, -np.infty)
  
  candidates = list(filter(lambda x: type(x) is not SpecialRoad, roads))
  if len(candidates) == 0: 
    return (None, -np.infty)
  isolation = {}
  diversity = {}
  for road in candidates:
    neighborTypes = {}
    isolation[road.id] = 0
    for neighb in chain(road.start.roads, road.end.roads):
      neighborTypes[str(type(neighb))] = neighborTypes.get(str(type(neighb)),0) + 1
      if roadClusters[neighb.id] == -1:
        isolation[road.id] += 1
      else:
        isolation[road.id] -= 0.5
    proportions = [ v / sum(neighborTypes.values()) for v in neighborTypes.values()]
    diversity[road.id] = sum( p * np.log(p) for p in proportions)
    
  scores = {}
  for road in candidates:
    scores[road.id] = isolation[road.id] * diversity[road.id] - road.level
  
  return sorted(scores.items(), key=lambda x: x[1])[-1]
  
def reskin(roads, resources):
  # A road will want to change type if
  # 1. Its resource production and demand are not being satisfied
  # 2. It does not match its neighbors
  
  candidates = [r for r in roads if type(r) is not TransportRoad and type(r) is not SpecialRoad]
  
  alignment = {} # positive is good
  for road in candidates:
    alignment[road.id] = 0
    for neighb in chain(road.start.roads, road.end.roads):
      if type(neighb) is TransportRoad:
        continue
      elif type(neighb) != type(road):
        alignment[road.id] -= 5
      elif road.level - neighb.level <= 1:
        alignment[road.id] += 1
      else:
        alignment[road.id] -= 0.5
  
  
  roadDemands = {} # positive is bad
  roadSupplies = {}
  for road in candidates:
    demands = {}
    supplies = {}
    for good, amts in road.baseProduction.items():
      supplies[good] = listAdd(supplies.get(good,[]), amts)
    for good, amts in road.baseDemand.items():
      demands[good] = listAdd(demands.get(good, []), amts)
    roadDemands[road.id] = demands
    roadSupplies[road.id] = supplies
  
  
  roadWasteSupply = {}
  for rID, supplies in roadSupplies.items():
    supply = {}
    for good, amts in supplies.items():
      badScore = listMultiply(amts, resources['production'].get(good, []))
      goodScore = listMultiply(amts, resources['demand'].get(good, []))
      
      supply[good] = listMinus(badScore, goodScore)
    roadWasteSupply[rID] = supply
  roadWasteDemand = {}        
  for rID, demands in roadDemands.items():
    demand = {}
    for good, amts in demands.items():
      # demand[good] = listDivide(listMinus(resources['production'].get(good, []), amts), normalizer[good])
      badScore = listMultiply(amts, resources['demand'].get(good, []))
      goodScore = listMultiply(amts, resources['production'].get(good, []))
      demand[good] = listMinus(badScore, goodScore)
    roadWasteDemand[rID] = demand
  
  scores = {}
  for road in candidates:
    resourceCount = sum(sum(amts) for amts in chain(roadWasteDemand[road.id].values(), roadWasteSupply[road.id].values()))
    scores[road.id] = resourceCount - 100 * alignment[road.id]
  
  toBeReskinned = sorted(scores.items(), key=lambda x: x[1])
  
  return toBeReskinned[-1], toBeReskinned[0]
  
def convert(road, newType):
  assert type(road) is not SpecialRoad, str(road) 
  if road.level > 1:
    road.levelDown()
  else:
    newRoad = newType(road.start, road.end)
    newRoad.add()
    road.remove()
  
  
if __name__ == '__main__':
   main()