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
  
  distance = np.linalg.solve( np.vstack([roadVector, roadPerp]).T, orig-dest)
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
    for i,(end1,end2) in enumerate(roads):
      result = roadDistance(node, allNodes[end1], allNodes[end2], roadSpeed)
      if result[1] < time:
        junct, time, isEndpoint = result
        roadIndex = i
    connectedToZero.add(nodeIndex)
    if junct is None:
      dirtRoads.append( (nodeIndex, destIndex))
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
  
  for a,b in majorNodes:
    x = scale*(a - upperLeft[0])
    y = scale*(b - upperLeft[1])
    draw.ellipse([(x-1,y-1), (x+1,y+1)], fill="blue")
  for a,b in minorNodes:
    x = scale*(a - upperLeft[0])
    y = scale*(b - upperLeft[1])
    draw.ellipse([(x-1,y-1), (x+1, y+1)], fill="red")
  
  img.save(fname)
  
  
def growCity(majorNodes, minorNodes, majorRoads, minorRoads, roadSpeed=10, iterations=10, nodePopulations=None, forcedMajor=set()):
  # idea will be that every major node gets 10 people, every minor node gets 2, and then each wants to go to another node with probability based on the population of that node
  # each intermediary gets half a pop for each pop that travels through
    
  allNodes = majorNodes + minorNodes
  allRoads = majorRoads + minorRoads
  
  adjacency, roadTraffic = makeAdjacency(allNodes, majorRoads, minorRoads, roadSpeed)
  
  
  for a, orig in enumerate(majorNodes):
    times, _ = dijkstra(a,allNodes, adjacency)
    for b, dest in enumerate(allNodes[a+1:], a+1):
      if a in adjacency[b]:
        continue
      
      bestTime = times[b] * 0.8
      bestCost = dist(orig,dest) + bestTime
      bestJunct = None
      bestRoad = None
      bestIsEndpoint = False
      for bRoad, bTime in adjacency[b].items():
        junct, time, isEndpoint = roadTo(orig, dest, allNodes[bRoad], bTime/dist(allNodes[bRoad], dest))
        fullTime = time + bTime * dist(junct,dest) / dist(allNodes[bRoad],dest)
        cost = 2 * fullTime - time
        if fullTime < bestTime:
          bestTime = fullTime
          bestJunct = junct
          bestRoad = bRoad
          bestIsEndpoint = isEndpoint
      if bestJunct is not None:
        print('Creating new route from %d to %d' % (a,b))
        if bestIsEndpoint:
          if dist(orig,dest) <= dist(orig,allNodes[bestRoad]):
            junctIndex = b
          else:
            junctIndex = bestRoad
        else:
        
          junctIndex = len(allNodes)
          allNodes.append(bestJunct)
          minorNodes.append(bestJunct)
          
          index = -1
          for i, road in enumerate(majorRoads):
            if b in road and bestRoad in road:
              index = i
          if index >= 0:
            majorRoads.pop(index)
            majorRoads.append((b, junctIndex))
            majorRoads.append((bestRoad, junctIndex))
          else:
            for i, road in enumerate(minorRoads):
              if b in road and bestRoad in road:
                index = i
            assert index >= 0
            minorRoads.pop(index)
            minorRoads.append((b, junctIndex))
            minorRoads.append((bestRoad, junctIndex))
          allRoads.append((b, junctIndex))
          allRoads.append((bestRoad, junctIndex))
          adjacency[junctIndex] = {}
          roadTraffic[junctIndex] = {}
                  
          oldTime = adjacency[b].pop(bestRoad)
          adjacency[bestRoad].pop(b)
          roadTraffic[b].pop(bestRoad)
          roadTraffic[bestRoad].pop(b)
          
          adjacency[b][junctIndex] = oldTime * dist(dest,bestJunct) / dist(dest,allNodes[bestRoad])
          adjacency[junctIndex][b] = adjacency[b][junctIndex]
          roadTraffic[b][junctIndex] = 0
          roadTraffic[junctIndex][b] = 0
          
          adjacency[bestRoad][junctIndex] = oldTime * dist(allNodes[bestRoad],bestJunct) / dist(dest,allNodes[bestRoad])
          adjacency[junctIndex][bestRoad] = adjacency[bestRoad][junctIndex]
          roadTraffic[bestRoad][junctIndex] = 0
          roadTraffic[junctIndex][bestRoad] = 0
          
        
        adjacency[a][junctIndex] = dist(orig, bestJunct) 
        adjacency[junctIndex][a] = adjacency[a][ junctIndex]
        roadTraffic[a][junctIndex] = 0
        roadTraffic[junctIndex][a] = 0
        minorRoads.append((a,junctIndex))
        allRoads.append((a, junctIndex))
        times, _ = dijkstra(a,allNodes, adjacency)
        
        checkIntersections(a, junctIndex, allNodes, adjacency, roadTraffic, nodePopulations)
        
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
  largeNodes = sorted(filter(lambda y: len(adjacency[y[0]]) < 5, enumerate(newPopulations)), key=lambda x: x[1], reverse=True)[:2]
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
    angle = angles[biggest] + size/2
    vector = np.array([np.cos(angle), np.sin(angle)])
    # idk put it the average distance to the nearest node far away
    scale = sum(dist(v,0) for v in vectors) / len(vectors)
    newNode = orig + scale * vector
    ct = sum( dist(n,newNode) < 20 for n in allNodes)
    while ct > 2:
      scale += 5
      newNode = orig + scale * vector
      ct = sum(dist(n, newNode) < 20 for n in allNodes)
    
    newPopulations.append(newPop)
    allNodes.append(newNode)
    #now to connect it
    nodeIndex = len(allNodes) -1
    minorRoads.append((largest[0], nodeIndex))
    allRoads.append(minorRoads[-1])
    adjacency[largest[0]][nodeIndex] = dist(orig, newNode)
    adjacency[nodeIndex] = {largest[0]: dist(orig, newNode)}
    roadTraffic[largest[0]][nodeIndex] = newPop
    roadTraffic[nodeIndex] = {largest[0]: newPop}
    checkIntersections(largest[0], nodeIndex, allNodes, adjacency, roadTraffic, newPopulations)
  else:
    placeRandom(allNodes, allRoads, majorRoads, minorRoads, adjacency, roadTraffic, roadSpeed)
  
  
  
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
  
  
  if len(forcedMajor) > 0:
    return majorNodes, minorNodes, majorRoads, minorRoads, nodePopulations, newForcedMajor
  else:
    return majorNodes, minorNodes, majorRoads, minorRoads, nodePopulations
      
def dijkstra(source, nodes, adjacency):
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
    print(prev)
    return distances, prev
    
def checkIntersections(end1, end2, allNodes, adjacency, roadTraffic, newPopulations, checked=None):
  node1 = allNodes[end1]
  node2 = allNodes[end2]
  if checked is None:
    checked = set()
  slope = node2 - node1
  slope = slope[1] / slope[0] if slope[0] != 0 else np.infty # won't actually support vertical lines
  inter = node2[1] - slope * node2[0]
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
      if (nodeA[0] - x)*(nodeB[0] -x) < 0 and (node1[0] - x)*(node2[0] - x) < 0:
        # they intersect
        print('Intersection found between %d,%d and %d,%d' % (end1, end2, a, b))
        newNode = np.array([x, slope*x + inter])
        nodeIndex = len(allNodes)
        allNodes.append(newNode)
        newPopulations.append(0)
        adjacency[nodeIndex] = {}
        roadTraffic[nodeIndex] = {}
        checked.add(nodeIndex)
        # split this road
        time = adjacency[end1].pop(end2)
        adjacency[end2].pop(end1)
        adjacency[end1][nodeIndex] = time * dist(node1, newNode) / dist(node1, node2)
        adjacency[nodeIndex][end1] = adjacency[end1][nodeIndex]
        adjacency[end2][nodeIndex] = time * dist(node2, newNode) / dist(node1, node2)
        adjacency[nodeIndex][end2] = adjacency[end2][nodeIndex]
        traffic = roadTraffic[end1].pop(end2)
        roadTraffic[end2].pop(end1)
        roadTraffic[end1][nodeIndex] = traffic
        roadTraffic[nodeIndex][end1] = roadTraffic[end1][nodeIndex]
        roadTraffic[end2][nodeIndex] = traffic
        roadTraffic[nodeIndex][end2] = roadTraffic[end2][nodeIndex]
        newPopulations[-1] += traffic / 2
        # split the other road
        
        time = adjacency[a].pop(b)
        adjacency[b].pop(a)
        adjacency[a][nodeIndex] = time * dist(nodeA, newNode) / dist(nodeA, nodeB)
        adjacency[nodeIndex][a] = adjacency[a][nodeIndex]
        adjacency[b][nodeIndex] = time * dist(nodeB, newNode) / dist(nodeA, nodeB)
        adjacency[nodeIndex][b] = adjacency[b][nodeIndex]
        traffic = roadTraffic[a].pop(b)
        roadTraffic[b].pop(a)
        roadTraffic[a][nodeIndex] = traffic
        roadTraffic[nodeIndex][a] = roadTraffic[a][nodeIndex]
        roadTraffic[b][nodeIndex] = traffic
        roadTraffic[nodeIndex][b] = roadTraffic[b][nodeIndex]
        newPopulations[-1] += traffic / 2
        # now need to split each of those halves
        checkIntersections(end1, nodeIndex, allNodes, adjacency, roadTraffic, newPopulations, checked)
        checkIntersections(end2, nodeIndex, allNodes, adjacency, roadTraffic, newPopulations, checked)
        return
      #else:
        #print('it wasn\'t')
        
        
          
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
    
    
if __name__ == '__main__':
  main()