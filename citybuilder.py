from PIL import Image, ImageDraw
import numpy as np
import random
from argparse import ArgumentParser


def options():
  parser = ArgumentParser()
  parser.add_argument('nodes', nargs='+')
  parser.add_argument('-s', '--speed', type=float, help='Speed on main roads', default=2)
  parser.add_argument('-n', '--iterations', type=int, default=10)
  parser.add_argument('-f', '--filename', default='test')
  parser.add_argument('-r', '--roads', nargs='+')
  parser.add_argument('-m', '--major', nargs='+', help='nodes to be forced to be major')

  return parser.parse_args()
  
def main():
  ops = options()
  nodes = [ np.array(list(map(int, n.split(',')))) for n in ops.nodes]
  roads = [ tuple(map(int, r.split(','))) for r in ops.roads ]
  major = set([int(n) for n in ops.major])
  majN, minN, majR, minR = connectWithoutNewRoads(nodes, roads, ops.speed)
  drawCity(majN, minN, majR, minR, '%s%02d.png' % (ops.filename, 0))
  
  majN, minN, majR, minR, pop, fM = growCity(majN, minN, majR, minR, roadSpeed=ops.speed, nodePopulations=None, forcedMajor=major)
  drawCity(majN, minN, majR, minR, '%s%02d.png' % (ops.filename, 1))
  
  for i in range(2,ops.iterations):
    majN, minN, majR, minR, pop, fM = growCity(majN, minN, majR, minR, roadSpeed=ops.speed, nodePopulations=pop, forcedMajor=fM)
    drawCity(majN, minN, majR, minR, '%s%02d.png' % (ops.filename, i))
  
  
def dist(a,b):
  return np.sqrt(np.square(a-b).sum())
  
def roadDistance(pt, a, b, roadSpeed): # a and b are endpoints of a road
  # first figure out which endpoint is closer
  if dist(a,pt) < dist(b,pt):
    near, far = a, b # wherever you are
  else:
    near, far = b, a
  # now want to make sure we are actually in between the endpoints, i.e. that we have an obtuse triangle
  adjPt = pt - near
  adjFar = far - near
  if np.dot(adjPt, adjFar) < 0:
    return None, np.infty # endpoints are closest
  
  
  
  # now to figure out where on the road we should connect, which means I need to know my distance to the close point in road and antiroad directions
  junction, timeToJunction, isEndpoint = roadTo(pt, near, far, roadSpeed)
  return junction, timeToJunction + dist(junction, near) / roadSpeed, isEndpoint
  
  
def roadTo(orig, dest, roadEndpoint, roadSpeed):
  roadVector = roadEndpoint - dest
  roadPerp = np.array([-roadVector[1], roadVector[0]])
  
  try:
    distance = np.linalg.solve( np.vstack([roadVector, roadPerp]).T, orig-dest)
  except Exception as e:
    e.args += (orig, dest, roadEndpoint)
    raise
  if distance[0] < 0:
    return dest, dist(orig, dest), True
  elif distance[0] > 1:
    return roadEndpoint, dist(orig, roadEndpoint), True
  elif (roadSpeed**2 - 1) < 1e-9:
    return dest, dist(orig, dest), True
  
  # can compute how far along the road from the endpoint to join
  
  junction = dest + (distance[0] - distance[1] / np.sqrt(roadSpeed**2 - 1)) * roadVector
  timeToJunction = dist(orig,junction)
  return junction, timeToJunction, False
    
  
def connectWithoutNewRoads(oldNodes, oldRoads, roadSpeed=5):
  # nodes is an array of np arrays representing the (x,y) coordinate of the node
  # roads is an array of tuples representing the nodes that are connected
  # this needs to identify what edges need to be added in order to connect the graph, and where those edges need to be (inserting nodes when needed)
  newNodes = []
  allNodes = [n for n in oldNodes]
  roads = [r for r in oldRoads]
  connectedToZero = {0}
  prevSize = -1
  while len(connectedToZero) != prevSize:
    prevSize = len(connectedToZero)
    for endpoints in roads:
      if len(connectedToZero.intersection(endpoints)) > 0:
        connectedToZero = connectedToZero.union(endpoints)
  
  unconnected = set([i for i in range(len(oldNodes)) if i not in connectedToZero])
  dirtRoads = []
  
  
  while len(unconnected) > 0:
    nodeIndex = unconnected.pop()
    node = allNodes[nodeIndex]
    # might connect to the closest node
    destIndex = sorted(connectedToZero, key=lambda x: dist(node,allNodes[x]))[0]
    dest = allNodes[destIndex]
    
    # likely will connect to an edge
    junct = None
    time = dist(node, dest)
    roadIndex = -1
    bestIsEndpoint = False
    for i,(end1,end2) in enumerate(roads):
      result = roadDistance(node, allNodes[end1], allNodes[end2], roadSpeed)
      if result[1] < time:
        junct, time, isEndpoint = result
        roadIndex = i
        bestIsEndpoint = isEndpoint
    connectedToZero.add(nodeIndex)
    if junct is None:
      dirtRoads.append( (nodeIndex, destIndex))
    elif bestIsEndpoint:
      if dirtRoads.append()
    else:
      junctIndex = len(allNodes)
      allNodes.append(junct)
      newNodes.append(junct)
      oldRoad = roads.pop(roadIndex)
      dirtRoads.append((nodeIndex, junctIndex))
      roads.append((oldRoad[0], junctIndex))
      roads.append((oldRoad[1], junctIndex))
      connectedToZero.add(junctIndex)
    
  return oldNodes, newNodes, roads, dirtRoads
  
  
  
def drawCity(majorNodes, minorNodes, majorRoads, minorRoads, fname):
  img = Image.new('RGBA', (400,400), (255,255,255,255))
  draw = ImageDraw.Draw(img)
  
  allNodes = majorNodes + minorNodes
  
  xMin = min(n[0] for n in allNodes) - 50
  yMin = min(n[1] for n in allNodes) - 50
  xMax = max( n[0] for n in allNodes) + 50
  yMax = max( n[1] for n in allNodes) + 50

  upperLeft = np.array([xMin, yMin])
  scale = max(1, 400/(xMax - xMin), 400/(yMax - yMin))
  
  
  
  for a,b in majorRoads:
    draw.line([tuple(scale*(allNodes[a]-upperLeft)), tuple(scale*(allNodes[b]-upperLeft))], fill="green")
  for a,b in minorRoads: 
    draw.line([tuple(scale*(allNodes[a]-upperLeft)), tuple(scale*(allNodes[b]-upperLeft))], fill="brown")
  
  for i,(a,b) in enumerate(majorNodes):
    x = scale*(a - upperLeft[0])
    y = scale*(b - upperLeft[1])
    draw.ellipse([(x-1,y-1), (x+1,y+1)], fill="blue")
    draw.text((x,y-10), str(i), fill='black')
  for i,(a,b) in enumerate(minorNodes, len(majorNodes)):
    x = scale*(a - upperLeft[0])
    y = scale*(b - upperLeft[1])
    draw.ellipse([(x-1,y-1), (x+1, y+1)], fill="red")
    draw.text((x,y-10), str(i), fill='black')
  img.save(fname)
  
  
def growCity(majorNodes, minorNodes, majorRoads, minorRoads, roadSpeed=10, iterations=10, nodePopulations=None, forcedMajor=set()):
  # idea will be that every major node gets 10 people, every minor node gets 2, and then each wants to go to another node with probability based on the population of that node
  # each intermediary gets half a pop for each pop that travels through
    
  allNodes = majorNodes + minorNodes
  allRoads = majorRoads + minorRoads
  
  adjacency, roadTraffic = makeAdjacency(allNodes, majorRoads, minorRoads, roadSpeed)
  
  assert len(roadTraffic) == len(adjacency)
  assert len(adjacency) == len(allNodes)
  assert min(len(a) for a in adjacency.values()) > 0
  assert 2*len(majorRoads) + 2*len(minorRoads) == sum(len(a) for a in adjacency.values())
  print('Entrance sanity checks passed')
  
  routeTime = np.infty
  overallCost = np.infty
  startChosen, endChosen = -1, -1
  junctChosen = None
  chosenIsEndpoint = False
  roadChosen = -1
  
  
  
  
  for a, orig in enumerate(allNodes):
    times, _ = dijkstra(a,allNodes, adjacency)
    for b, dest in enumerate(allNodes[a+1:], a+1):
      if a in adjacency[b]:
        continue
      
      bestTime = times[b]
      bestCost = np.infty
      bestJunct = None
      bestRoad = None
      bestIsEndpoint = False
      for bRoad, bTime in adjacency[b].items():
        junct, time, isEndpoint = roadTo(orig, dest, allNodes[bRoad], bTime/dist(allNodes[bRoad], dest))
        fullTime = time + bTime * dist(junct,dest) / dist(allNodes[bRoad],dest)
        cost = time
        if fullTime < bestTime:
          bestTime = fullTime
          bestJunct = junct
          bestCost = cost
          bestRoad = bRoad
          bestIsEndpoint = isEndpoint
      if bestJunct is not None:
        if bestCost < overallCost:
          print('Connecting %d to %d would be cheap: %f' % (a,b, bestCost))
          startChosen, endChosen = a, b
          junctChosen = junct
          routeTime = fullTime
          chosenIsEndpoint = bestIsEndpoint
          roadChosen = bestRoad
          overallCost = bestCost
      
  assert len(adjacency) == len(allNodes)
  if startChosen > -1:
    orig = allNodes[startChosen]
    dest = allNodes[endChosen]
    print('Creating new route from %d to %d' % (startChosen, endChosen))
    if chosenIsEndpoint:
      if dist(orig,dest) <= dist(orig,allNodes[roadChosen]):
        junctIndex = endChosen
      else:
        junctIndex = roadChosen
    else:
    
      junctIndex = len(allNodes)
      allNodes.append(junctChosen)
      
      
      # adjacency[junctIndex] = {}
      #roadTraffic[junctIndex] = {}
              
      # oldTime = adjacency[endChosen].pop(roadChosen)
      # adjacency[roadChosen].pop(b)
      # roadTraffic[endChosen].pop(roadChosen)
      # roadTraffic[roadChosen].pop(b)
      
      # adjacency[endChosen][junctIndex] = oldTime * dist(dest,junctChosen) / dist(dest,allNodes[roadChosen])
      # adjacency[junctIndex][endChosen] = adjacency[endChosen][junctIndex]
      # roadTraffic[endChosen][junctIndex] = 0
      # roadTraffic[junctIndex][endChosen] = 0
      
      # adjacency[roadChosen][junctIndex] = oldTime * dist(allNodes[roadChosen],junctChosen) / dist(dest,allNodes[roadChosen])
      # adjacency[junctIndex][roadChosen] = adjacency[roadChosen][junctIndex]
      # roadTraffic[roadChosen][junctIndex] = 0
      # roadTraffic[junctIndex][roadChosen] = 0
    
    
    # roadTraffic[startChosen][junctIndex] = 0
    # roadTraffic[junctIndex][startChosen] = 0
    
    minorRoads.append((startChosen,junctIndex))
    
    allRoads.append((startChosen, junctIndex))
    
    
    junctIndex = checkIntersections(startChosen, junctIndex, allNodes, adjacency, roadTraffic, nodePopulations)
  
    if not chosenIsEndpoint:
      index = -1
      for i, road in enumerate(majorRoads):
        if endChosen in road and roadChosen in road:
          index = i
      if index >= 0:
        majorRoads.pop(index)
        majorRoads.append((endChosen, junctIndex))
        majorRoads.append((roadChosen, junctIndex))
      else:
        for i, road in enumerate(minorRoads):
          if b in road and roadChosen in road:
            index = i
        assert index >= 0
        minorRoads.pop(index)
        minorRoads.append((endChosen, junctIndex))
        minorRoads.append((roadChosen, junctIndex))
    allRoads.append((endChosen, junctIndex))
    allRoads.append((roadChosen, junctIndex))
    minorNodes.append(allNodes[junctIndex])
    # adjacency[startChosen][junctIndex] = dist(orig, junctChosen) 
    # adjacency[junctIndex][startChosen] = adjacency[startChosen][ junctIndex]
    
  assert len(adjacency) == len(allNodes)  
  
  if nodePopulations is None:
    nodePopulations = [10] * len(majorNodes) + [2] * len(minorNodes)
  population = sum(nodePopulations)
  newPopulations = [0] * len(allNodes)
  traffic = 0
  # this means we're going to need Dijkstra
  for i,node in enumerate(allNodes):
    p = nodePopulations[i]
    dists, prev = dijkstra(i,allNodes, adjacency)
    normalizer = sum( nodePopulations[j] / (dists[j] * population) for j in range(len(allNodes)) if j != i)
    for j, dest in enumerate(allNodes):
      if i == j:
        continue
      travelers = nodePopulations[j] * p / (dists[j] * population * normalizer)
      newPopulations[j] += travelers
      layover = j
      while prev[layover] != i:
        roadTraffic[layover][prev[layover]] += travelers
        roadTraffic[prev[layover]][layover] += travelers
        traffic += travelers
        layover = prev[layover]
        newPopulations[layover] += travelers/2
  
  # Finally, the largest node (with fewer than five children) splits off its population to make a nearby minor node
  largeNodes = sorted(filter(lambda y: len(adjacency[y[0]]) < 5 and len(adjacency[y[0]]) > 2, enumerate(newPopulations)), key=lambda x: x[1], reverse=True)[:2]
  if len(largeNodes) > 0:
    if len(largeNodes) == 2:
      largest, nextLargest = largeNodes
      newPop = min(nextLargest[1], largest[1]- nextLargest[1])
      print("Largest node has %f pop, making node with %f pop" % (largest[1], newPop))
    else:
      largest = largeNodes[0]
      newPop = largest[1] / 2
    newPopulations[largest[0]] -= newPop
    orig = allNodes[largest[0]]
    vectors = [ allNodes[i] - orig for i in adjacency[largest[0]]]
    dotprods = [(v, np.dot([1,0], v/dist(v,0))) for v in vectors]
    angles = sorted([np.arccos(d) if np.dot([0,1], v) > 0 else (2*np.pi - np.arccos(d)) for v,d in dotprods])
    intervals = [angles[i] - angles[i-1] for i in range(1,len(angles))]
    intervals.append( 2*np.pi - angles[-1] + angles[0])
    biggest, size = sorted(enumerate(intervals), key=lambda x: x[1], reverse=True)[0]
    angle = angles[biggest] + size*((1+random.random())/3)
    vector = np.array([np.cos(angle), np.sin(angle)])
    # idk put it the average distance to the nearest node far away
    scale = sum(dist(v,0) for v in vectors) / len(vectors)
    newNode = orig + scale * vector
    ct = sum( dist(n,newNode) < 10 for n in allNodes)
    while ct > 0:
      scale += 5
      newNode = orig + scale * vector
      ct = sum(dist(n, newNode) < 20 for n in allNodes)
    
    newPopulations.append(newPop)
    allNodes.append(newNode)
    #now to connect it
    print('New node will be at (%f,%f)' % (newNode[0], newNode[1]))
    nodeIndex = len(allNodes) -1
    otherEndpoint = checkIntersections(nodeIndex, largest[0], allNodes, adjacency, roadTraffic, newPopulations)
    
    minorRoads.append((otherEndpoint, nodeIndex))
    allRoads.append(minorRoads[-1])
    
    # roadTraffic[largest[0]][nodeIndex] = newPop
    # roadTraffic[nodeIndex] = {largest[0]: newPop}
    
  else:
    placeRandom(allNodes, allRoads, majorRoads, minorRoads, adjacency, roadTraffic, roadSpeed)
  
  
  cleanMap(allNodes, newPopulations, roadTraffic)
  
  
  majorNodes = []
  minorNodes = []
  majorRoads = []
  minorRoads = []
  updateMajor = {}
  updateMinor = {}
  
  for i,p in enumerate(newPopulations):
    if p >= population / len(newPopulations) or i in forcedMajor:
      updateMajor[i] = len(majorNodes)
      majorNodes.append(allNodes[i])
      if i in forcedMajor:
        print('Has a population of %f, needs to be at least %f to ride' % (p, population/len(newPopulations)))
      newPopulations[i] = max(p, population/len(newPopulations))
    else:
      updateMinor[i] = len(minorNodes)
      minorNodes.append(allNodes[i])
    
  update = {i: j for i,j in updateMajor.items()}
  for i,j in updateMinor.items():
    update[i] = j + len(updateMajor)
  for i in range(len(update),len(allNodes)):
    update[i] = i
  allNodes = majorNodes + minorNodes
  for a in roadTraffic.keys():
    newA = update[a]
    for b, t in roadTraffic[a].items():
      newB = update[b]
      if newA < newB:
        if t > traffic / len(allRoads):
          majorRoads.append( (newA, newB))
        else:
          minorRoads.append( (newA, newB))
          
  print(update)
  newForcedMajor = set([update[i] for i in forcedMajor])
  nodePopulations = [0] * len(newPopulations)
  for i,j in update.items():
    nodePopulations[j] = newPopulations[i]
  
  assert len(newPopulations) == len(allNodes)
  assert len(roadTraffic) == len(allNodes)
  assert min(len(a) for a in roadTraffic.values()) > 0
  try:
    assert 2*len(majorRoads) + 2*len(minorRoads) == sum(len(a) for a in roadTraffic.values())
  except AssertionError as e:
    e.args += ('Major roads: %d, Minor roads: %d, Road Traffic: %d' % (len(majorRoads), len(minorRoads), sum(len(a) for a in roadTraffic.values())),majorRoads, minorRoads, '\n'.join('%d: %s' % (i, ', '.join(str(s) for s in roadTraffic[i].keys())) for i in roadTraffic))
    raise
  print('Exit sanity checks passed')
  
  if len(forcedMajor) > 0:
    return majorNodes, minorNodes, majorRoads, minorRoads, nodePopulations, newForcedMajor
  else:
    return majorNodes, minorNodes, majorRoads, minorRoads, nodePopulations
      
def dijkstra(source, nodes, adjacency):
#    print('\n'.join('%d: %s' % (i, ', '.join(str(s) for s in adjacency[i].keys())) for i in adjacency))
    
    distances = [np.infty] * len(nodes)
    prev = [-1] * len(nodes)
    Q = set(range(len(nodes)))
    distances[source] = 0
    while len(Q) > 0:
      u = sorted(Q, key=lambda x: distances[x])[0]
      Q.remove(u)
      for neighb,distance in adjacency[u].items():
        alt = distances[u] + distance
        if alt < distances[neighb]:
          distances[neighb] = alt
          prev[neighb] = u
#    print(prev)
    return distances, prev
    
def checkIntersections(end1, end2, allNodes, adjacency, roadTraffic, newPopulations, checked=None): # end1 is the existing node, end2 is the proposed node
  print('Checking intersections between %d and %d' % (end1, end2))
  node1 = allNodes[end1]
  node2 = allNodes[end2]
  if checked is None:
    checked = set()
  slope = node2 - node1
  slope = slope[1] / slope[0] if slope[0] != 0 else np.infty # won't actually support vertical lines
  inter = node2[1] - slope * node2[0]
  
  closestIntersection = np.infty
  closestIntersectionPoint = -1
  closestEndpoints = (-1,-1)
  
  newPop = 0 if newPopulations is None else newPopulations[-1]
  
  
  
  for a in adjacency:
    if a in checked or a in [end1, end2]:
      continue
    checked.add(a)
    nodeA = allNodes[a]
    for b in adjacency[a]:
      if b in [end1, end2]:
        continue
      nodeB = allNodes[b]
      otherSlope = nodeB - nodeA
      otherSlope = otherSlope[1] / otherSlope[0] if otherSlope[0] != 0 else np.infty
      otherInter = nodeB[1] - otherSlope * nodeB[0]
      x = - (otherInter - inter) / (otherSlope - slope) if otherSlope - slope != 0 else np.infty # parallel shouldn't be an issue
      #print('Checking whether %f is in [%f, %f] and [%f,%f]' % (x, nodeA[0], nodeB[0], node1[0], node2[0]))
      if (nodeA[0] - x)*(nodeB[0] -x) <= 1e-11 and (node1[0] - x)*(node2[0] - x) < 1e-11:
        # they intersect
        print('Intersection found between %d,%d and %d,%d' % (end1, end2, a, b))
        if abs(node1[0] - x) < closestIntersection:
          closestIntersection = abs(node1[0]- x)
          closestIntersectionPoint = x
          closestEndpoints = (a,b)
  if closestIntersection < np.infty:
    if len(adjacency.get(end2,{})) > 0:
      newPopulations.append(0)
    else:
      allNodes.pop(-1)
      
    end2 = len(allNodes)
    adjacency[end2] = adjacency.get(end2,{})
    roadTraffic[end2] = roadTraffic.get(end2, {})
    print('Edge will go from %d to %d' % (end1, end2))
    a,b = closestEndpoints
    nodeA, nodeB = allNodes[a], allNodes[b]
    x = closestIntersectionPoint

    newNode = np.array([x, slope*x + inter])
    allNodes.append(newNode)
    print('Placing node on interection point: (%f, %f)' % (newNode[0], newNode[1]))
    checked.add(end2)
    # split the intersection
    
    
    
    time = adjacency[a].pop(b)
    adjacency[b].pop(a)
    adjacency[a][end2] = time * dist(nodeA, newNode) / dist(nodeA, nodeB)
    adjacency[end2][a] = adjacency[a][end2]
    adjacency[b][end2] = time * dist(nodeB, newNode) / dist(nodeA, nodeB)
    adjacency[end2][b] = adjacency[b][end2]
    traffic = roadTraffic[a].pop(b)
    roadTraffic[b].pop(a)
    roadTraffic[a][end2] = traffic
    roadTraffic[end2][a] = roadTraffic[a][end2]
    roadTraffic[b][end2] = traffic
    roadTraffic[end2][b] = roadTraffic[b][end2]
    newPopulations[-1] += traffic / 2
    
  else:
    print('No intersection, hooking up node %d' % end2)
    newNode = allNodes[-1]
  # assign the intended road
  time = dist(node1, newNode)
  if end1 not in adjacency:
    adjacency[end1] = {}
    roadTraffic[end1] = {}
  adjacency[end1][end2] = time
  adjacency[end2][end1] = adjacency[end1][end2]

  roadTraffic[end1][end2] = newPop
  roadTraffic[end2][end1] = roadTraffic[end1][end2]

  assert min(len(a) for a in adjacency.values()) > 0
  return end2
        
          
def makeAdjacency(allNodes, majorRoads, minorRoads, roadSpeed):
    roadTraffic = {i: {} for i in range(len(allNodes))}
    adjacency = {i: {} for i in range(len(allNodes))}
    for a,b in majorRoads:
      adjacency[a][b] = dist(allNodes[a], allNodes[b]) / roadSpeed
      adjacency[b][a] = adjacency[a][b]
      roadTraffic[a][b] = 0
      roadTraffic[b][a] = 0
    for a,b in minorRoads:
      adjacency[a][b] = dist(allNodes[a], allNodes[b])
      adjacency[b][a] = adjacency[a][b]
      roadTraffic[a][b] = 0
      roadTraffic[b][a] = 0
    
    return adjacency, roadTraffic
    
    
def placeRandom(allNodes, allRoads, majorRoads, minorRoads, adjacency, roadTraffic, roadSpeed):
      xMin = min(n[0] for n in allNodes)
      yMin = min(n[1] for n in allNodes)
      xRange = max( n[0] for n in allNodes) - xMin
      yRange = max( n[1] for n in allNodes) - yMin
      
      nearCount = 100
      while nearCount > 5:
        xOffset = random.random() * xRange * 1.5
        yOffset = random.random() * yRange * 1.5
        newNode = np.array([xMin + xOffset - xRange / 4, yMin + yOffset - yRange/4])
        nearCount = sum( dist(newNode,n) < 30 for n in allNodes)
      
      allNodes.append(newNode)
      
      _, newNodes, oldRoads, newRoads = connectWithoutNewRoads(allNodes, allRoads, roadSpeed)
      assert len(newRoads) == 1
      if len(newNodes) > 0:
        assert len(newNodes) == 1
        nodeIndex = len(allNodes)
        fragments = filter(lambda x: nodeIndex in x, oldRoads)
        endpoints = tuple([ min(f) for f in fragments])
        if endpoints in majorRoads:
          roadList = majorRoads
          a, b = endpoints
        elif endpoints[::-1] in majorRoads:
          roadList = majorRoads
          b, a = endpoints
        elif endpoints in minorRoads:
          roadList = minorRoads
          a, b = endpoints
        else:
          assert endpoints[::-1] in minorRoads
          roadList = minorRoads
          b, a = endpoints
        roadList.remove((a,b))
        roadList.add((a, nodeIndex))
        roadList.add((b, nodeIndex))
        roadTraffic[a][nodeIndex] = roadTraffic[a].pop(b)
        roadTraffic[nodeIndex] = {a: roadTraffic[a][nodeIndex]}
        roadTraffic[b][nodeIndex] = roadTraffic[b].pop(a)
        roadTraffic[nodeIndex] = {b: roadTraffic[b][nodeIndex]}
          
        allNodes.append(newNodes[0])
     
      minorRoads.append(newRoads[0])
      a, b = sorted(newRoads[0])
      roadTraffic[a][b] = newPop
      roadTraffic[b] = {a: newPop}
      
      adjacency, _ = makeAdjacency(allNodes, majorRoads, minorRoads, roadSpeed)


def cleanMap(allNodes, newPopulations, roadTraffic, threshold=5):
  # if there are two nodes that are really close together, merge them
  i = 0
  while i < len(allNodes):
    j = i+1
    while j < len(allNodes):
      if dist(allNodes[i], allNodes[j]) > threshold:
        j += 1
        continue
      print('Merging %d into %d' % (j,i))
      node = (allNodes[i] + allNodes[j]) / 2
      pop = newPopulations[i] + newPopulations[j]
      
      for road, traffic in roadTraffic[j].items():
        if road == i:
          continue
        roadTraffic[i][road] = roadTraffic[i].get(road, 0) + traffic
        roadTraffic[road][i] = roadTraffic[i][road]
      
      allNodes.pop(j)
      newPopulations.pop(j)
      roadTraffic.pop(j)
      for road in roadTraffic.values():
        if j in road:
          road.pop(j)

      # now need to decrement every number greater than j that appears
      for k in range(len(allNodes)):
        if k >= j:
          roadTraffic[k] = roadTraffic.pop(k+1)
        for l in range(j, len(allNodes)):
          if l+1 in roadTraffic[k]:
            roadTraffic[k][l] = roadTraffic[k].pop(l+1)
          
      allNodes[i] = node
      newPopulations[i] = pop
    i += 1
  print('%d nodes remaining' % len(allNodes))
if __name__ == '__main__':
  main()