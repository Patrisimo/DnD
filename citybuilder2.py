import numpy as np
from itertools import chain
from roadsnodes import *
import random
from PIL import Image, ImageDraw
import logging
from argparse import ArgumentParser
import json

logging.basicConfig(level=logging.INFO)

def options():
  parser = ArgumentParser()
  
  parser.add_argument('iterations', type=int)
  parser.add_argument('--load')
  
  
  return parser.parse_args()

def main():
  ops = options()
  
  if ops.load is None:
    FILENAME = 'newtest'
    a = Node(0,0)
    b = Node(40, 0)
    c = Node(10, 10)
    d = Node(100, 90)
    a.add()
    b.add()
    c.add()
    d.add()
    
    assert a.isActive()
    
    
    X = SpecialRoad(a,c)
    X.create({GoodType.MATERIAL: 20}, {GoodType.GOOD: 20}, 1, 1, 'purple')
    X.add()
    Y = TransportRoad(c,d)
    Y.add()
    
    assert Y in d.roads
    startIter = 0
    drawCity('%s.png' % (FILENAME))
  else:
    with open(ops.load) as file:
      data = json.load(file)
    FILENAME = data['filename']
    startIter = data['iter']
    for node in data['nodes']:
      Node(node['coord'], id=node['id'])
    Node.nodeId = max(Node.nodeSet.keys())+1
    
    for road in data['roads']:
      newRoad = Road.getClass(road['type'])(Node.nodeSet[road['start']], Node.nodeSet[road['end']], level=road['level'], id=road['id'])
      if type(newRoad) is SpecialRoad:
        newRoad.create(**road['data'])
      newRoad.reset()
    Road.roadId = max(Road.roadSet.keys())+1  
  for i in range(startIter, startIter + ops.iterations):
    cycle()
    drawCity('%s%02d.png' % (FILENAME, i))

  drawCity('%s_final.png' % FILENAME, clean=True)
  nodeInfo = [{'coord': n.coord.tolist(), 'id': n.id, 'roads': [r.id for r in n.roads]} for n in Node.nodeSet.values()]
  roadInfo = [{'start': r.start.id, 'end': r.end.id, 'type': str(type(r)), 'level': r.level, 'data': r.data if type(r) is SpecialRoad else None, 'id': r.id} for r in Road.roadSet.values()]
    
  with open('%s.json' % FILENAME, 'w') as file:
    json.dump({'nodes': nodeInfo, 'roads': roadInfo, 'filename': FILENAME, 'iter': ops.iterations+startIter}, file)
    
def cycle():
  # First, look for two nodes that are close together but not connected 
  # Say, take the pair with the smallest dist(a,b) / roadDist(a,b)
  
  cleanNodes(Node.nodeSet)
  cleanRoads(Road.roadSet)
  split(Road.roadSet)
  
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
    
  # Now do production
  # Each edge produces an amount depending on its length and level
  # This needs to be made available to every other edge
  productMultiplier = {} # edge -> source -> multiplier
  productProduced = {} # source -> product
  amountSupplied = 0
  
  for road in Road.roadSet.values():
    productMultiplier[road.id] = dijkstraMultiplier(road, Road.roadSet)
    productProduced[road.id] = road.produce()
    #logging.info('%s: %s' % (str(road), str(productMultiplier[road.id])))
    
  
  # Now do demand
  # Each edge demands an amount depending on its length and level
  # 
  #logging.info(productProduced)
  production = {}

  for n, goods in productProduced.items():
    for g, amts in goods.items():
      #logging.info('%s produces %s %s' % (str(Road.roadSet[n]), str(amts), str(g)))
      production[g] = listAdd(production.get(g,[]) ,amts)
  logging.info('Production:')
  for g in production:
    logging.info('%s: %s' % (str(g), str(production[g])))
  
  
  demand = {}
  for road in Road.roadSet.values():
    road.reset()
    for g, amts in road.demand().items():
      #logging.info('%s demands %s %s' % (str(road), str(amts), str(g)))
      demand[g] = listAdd(demand.get(g, []),amts)
  logging.info('Demand')
  for g in demand:
    logging.info('%s: %s' % (str(g), str(demand[g])))
  
  amountSupplied = sum(sum(sum(q) for q in p.values()) for p in productProduced.values())
  
  while amountSupplied > 0:
    # Prioritize delivering to high-level edges first
    # Fill the largest possible orders first
    orderSize = 0
    orderData = {'source': None, 'target': None, 'type': None, 'mult': 0, 'level': 0}
    # Find the connection with the least loss
    
    
    for road in Road.roadSet.values():
      demands = road.demand() 
      for source, products in productProduced.items():
        mult = productMultiplier[road.id]['mult'][source]
        prev = productMultiplier[road.id]['prev']
        for good, amts in demands.items():
          for level, amt in enumerate(amts):
            if good in products and level < len(products[good]):
              size = min(products[good][level]*mult, amt)
              assert size >= 0, (products, mult, amt)
              if orderSize == 0 or (road.level >= orderData['target'].level and (1-mult) * size <= (1-orderData['mult'])*orderSize and size > 0):
                orderSize = size
                orderData['source'] = source
                orderData['target'] = road
                orderData['type'] = good
                orderData['mult'] = mult
                orderData['level'] = level
                orderData['prev'] = prev
                assert size / mult <= products[good][level] + 1e-5
              elif size > 0:
                assert orderSize > 0, (orderSize, orderData, str(road), size, mult)
    if orderSize > 0:      
      #logging.info('Fulfilling order of level %d %f %s from %s to %s, only %f will arrive' % (orderData['level'], orderSize / orderData['mult'], str(orderData['type']), str(Road.roadSet[orderData['source']]), str(orderData['target']), orderSize ))
      path = orderData['source']
      size = orderSize 
      while orderData['prev'][path] > -1:
        path = orderData['prev'][path]
        road = Road.roadSet[path]
        size /= road.transportMultiplier()
        road.transit[orderData['type']] = road.transit.get(orderData['type'],0) + size
      
      productProduced[orderData['source']][orderData['type']][orderData['level']] -= orderSize / orderData['mult']
      assert productProduced[orderData['source']][orderData['type']][orderData['level']] > -1e-5
      if productProduced[orderData['source']][orderData['type']][orderData['level']] < 0:
        productProduced[orderData['source']][orderData['type']][orderData['level']] = 0
      #logging.info(orderData['target'].supplies)
      orderData['target'].supplies[orderData['type']][orderData['level']] -= orderSize 
      #logging.info(orderData['target'].supplies)
      amountSupplied -= orderSize * orderData['mult']
      production[orderData['type']] -= orderSize * orderData['mult']
    else:
      logging.info('Orders stopped because best order was size 0')
      break
  else:
    logging.info('Orders stopped because production calculated to be exhausted')
  production = {}

  logging.info('Left over data')
  for n, goods in productProduced.items():
    for g, amt in goods.items():
      #logging.info('%s produced extra %s %s' % (str(Road.roadSet[n]), str(amt), str(g)))
      production[g] = listAdd(production.get(g,[]), amt)
  logging.info('Extra Production:')
  for g in production:
    logging.info('%s: %s' % (str(g), str(production[g])))
  
  
  demand = {}
  for road in Road.roadSet.values():
    for g, amt in road.demand().items():
      #logging.info('%s still demands %s %s' % (str(road), str(amt), str(g)))
      demand[g] = listAdd(demand.get(g, []),amt)
  logging.info('Unmet Demand')
  for g in demand:
    logging.info('%s: %s' % (str(g), str(demand[g])))
  
  for g in production:
    assert listDot(demand.get(g,[]), production.get(g,0)) == 0, productProduced
  
  
  #logging.info('Checking residual demand')
  limitingResources = {}
  for road in Road.roadSet.values():
    #logging.info('%s: %s' % (str(road), str(road.supplies)))
    for good, amts in road.supplies.items():
      limitingResources[good] = limitingResources.get(good,0) - amts[0]


  for good, amts in production.items():
    limitingResources[good] = limitingResources.get(good,0) + amts[0]
  
    # Now level up all the roads
  for road in Road.roadSet.values():
    road.setLevel()
    road.reset()
  
  bestGood, amt = sorted(limitingResources.items(), key=lambda x: abs(x[1]))[-1]
  logging.info('Remaining goods: %s, %f' % (str(bestGood), amt))
  if abs(amt) > 0:
    if amt < 0:
      roadType = GoodType.roadSupply(bestGood)
    else:
      roadType = GoodType.roadDemand(bestGood)
    
    if random.random() > 0:
      createNewRoad(roadType, abs(amt))
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
  cleanNodes(Node.nodeSet)
  cleanRoads(Road.roadSet)
      
def findClosePair(nodeSet, roadSet): # find the node and the (node or edge) that are clsoest together
  logging.info('Finding closest pair of nodes')
  closePair = (None, None)
  closeRelativeDistance = np.infty
  closeDirectDistance = np.infty
  for id1, node1 in nodeSet.items():
    distances, prev = dijkstra(id1, nodeSet)
    logging.info(str(distances))
    for id2, node2 in nodeSet.items(): # find the closest node
      if id2 == id1:
        continue
      if min([abs(angle(node1, node2) - r.getAngle()) for r in node2.roads], default=np.infty) < 0.7:
        #logging.info('%s is too collinear with a road coming out of %s' % (node1, node2))
        continue
      directDistance = dist(node1, node2)
      relativeDistance = directDistance / distances[node2.id]
      if relativeDistance +1e-9 < closeRelativeDistance:
        logging.info('Found a close pair: %s is close to %s' % (str(node1), str(node2)))
        closeRelativeDistance = relativeDistance
        closeDirectDistance = directDistance
        closePair = (node1, node2)
      elif relativeDistance == closeRelativeDistance and directDistance + 1e-9< closeDirectDistance:
        logging.info('Found a more direct close pair: %s is close to %s' % (str(node1), str(node2)))
        closeRelativeDistance = relativeDistance
        closeDirectDistance = directDistance
        closePair = (node1, node2)
    
    for id2, road in roadSet.items():
      pt1, pt2 = road.endpoints
      
      if road in node1.roads:
        continue
      #logging.info('Road %s not connected to node %s: %s' % (str(road), str(node1), ', '.join(str(r) for r in node1.roads)))
      if angle(pt1, pt2, node1.coord)*2 < np.pi and angle(pt2, pt1, node1.coord)*2 < np.pi: # we're in between, need to find point of intersection
        inter, directDistance, closeEndpoint = findIntersection(road, node1)
        normalDistance = distances[closeEndpoint.id] + dist(closeEndpoint, inter) * road.travelTime() / road.length
        relativeDistance = directDistance / normalDistance
        if abs(angle(node1, inter) - road.getAngle()) < 0.1:
          continue
        elif relativeDistance + 1e-9 < closeRelativeDistance:
          logging.info('Found a close pair: %s is close to road %s at %s' % (str(node1), str(road), str(inter)))
          closeRelativeDistance = relativeDistance
          closeDirectDistance = directDistance
          closePair = (node1, inter)
        elif relativeDistance == closeRelativeDistance and directDistance + 1e-9 < closeDirectDistance:
          logging.info('Found a more direct close pair: %s is close to road %s at %s' % (str(node1), str(road), str(inter)))
          closeRelativeDistance = relativeDistance
          closeDirectDistance = directDistance
          closePair = (node1, inter)
    
        
  logging.info('Closest pair is %s to %s, with a relative distance of %5f and direct distance of %5f' % (str(closePair[0]), str(closePair[1]), closeRelativeDistance, closeDirectDistance))  
  return closePair, closeRelativeDistance
  
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
  if angle1 > 0 and angle1*2 < np.pi and angle2 > 0 and angle2*2 < np.pi: # find point of intersection
    grad = pt2.coord - pt1.coord
    perp = np.array([-grad[1], grad[0]])
    t1, t2 = -np.linalg.solve( np.vstack([grad, perp]).T, pt1.coord - pt.coord)
    assert t1 >= 0, (t1, str(pt1), str(pt2), str(pt))
    assert t1 <= 1, (t1, str(pt1), str(pt2), str(pt))
    assert abs(t2) > 1e-14, (t1, str(pt1), str(pt2), str(pt))
    if t1 * road.length < 10:
      logging.info('Too close to start node')
      return pt1, dist(pt1, pt), pt1
    elif (1-t1) * road.length < 10:
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
  grad1 = end2.coord - end1.coord
  
  pt1, pt2 = road.start, road.end
  grad2 = pt2.coord - pt1.coord
  
  t1, t2 = np.linalg.solve(np.vstack([grad1, -grad2]).T, pt1.coord - end1.coord)

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
  assert abs((newStart.coord - newEnd.coord).sum()) > 0, (str(newStart), str(newEnd))
  road = roadType(newStart, newEnd)
  if road.length > 5:
    road.add()
  else:
    logging.info('Resultant road too short')
  return road
  
def drawCity(fname, clean=False):
  logging.info('Saving %s' % fname)
  img = Image.new('RGBA', (400,400), (255,255,255,255))
  draw = ImageDraw.Draw(img)
  
  xMin = min(n.coord[0] for n in Node.nodeSet.values()) - 50
  yMin = min(n.coord[1] for n in Node.nodeSet.values()) - 50
  xMax = max(n.coord[0] for n in Node.nodeSet.values()) + 50
  yMax = max(n.coord[1] for n in Node.nodeSet.values()) + 50

  upperLeft = np.array([xMin, yMin])
  scale = min(1, 400/(xMax - xMin), 400/(yMax - yMin))
  
  
  logging.info('Roads:')
  for road in Road.roadSet.values():
    logging.info(str(road))
    draw.line([tuple(scale*(road.start.coord-upperLeft)), tuple(scale*(road.end.coord-upperLeft))], fill=road.color(), width=int(np.sqrt(2*road.level)))
  
  logging.info('\nNodes:')
  for n in Node.nodeSet.values():
    logging.info(str(n))
    x = scale*(n.coord[0] - upperLeft[0])
    y = scale*(n.coord[1] - upperLeft[1])
    draw.ellipse([(x-1,y-1), (x+1,y+1)], fill="blue")
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
      if dist(node1, node2) < 5:
        # need to combine
        nodeList.remove(node2.id)
        node2.replace(node1)
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
        

      
def createNewRoad(roadType, amt):  
# randomly choose a node with probability proportional to how sparse it is
  newLevel = 1
  amt = min(amt, 40)
  logging.info('Creating new %s of length %.3f' % (str(roadType), amt))
  
  probs = {}
  for node in Node.nodeSet.values():
    if len(node.roads) < 5:
      probs[node.id] = 1. / len(node.roads)

  assert len(probs) > 0, '\n'.join([': '.join((str(n), str(len(n.roads)), '\n'.join(map(str, n.roads)))) for n in Node.nodeSet.values()])
  choice = random.random() * sum(v for v in probs.values())
  assert choice > 0, probs
  luckyNode = -1
  while choice > 0:
    luckyNode, weight = probs.popitem()
    choice -= weight
  
  newStart = node.nodeSet[luckyNode]
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
      

if __name__ == '__main__':
   main()