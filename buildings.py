import numpy as np
import json
from roadsnodes import *
from quadtree import QuadTree
from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageColor
from quadtreeTest import draw as qtDraw

# logging.disable(logging.CRITICAL)

def load(filename, zoom):
  with open(filename) as file:
    data = json.load(file)
  FILENAME = data['filename']
  startIter = data['iter']
  
  tree = QuadTree( (data['treeMin'][0]*zoom, data['treeMin'][1]*zoom), (data['treeMax'][0]*zoom, data['treeMax'][1]*zoom))
  Node.tree = tree
  Road.tree = tree
  for node in data['nodes']:
    Node((node['coord'][0]*zoom, node['coord'][1]*zoom), id=node['id'])
  Node.nodeId = max(Node.nodeSet.keys())+1
  
  for road in data['roads']:
    newRoad = Road.getClass(road['type'])(Node.nodeSet[road['start']], Node.nodeSet[road['end']], level=road['level'])
    if type(newRoad) is SpecialRoad:
      newRoad.create(**road['data'])
    newRoad.add(road['id'])
    newRoad.reset()
  Road.roadId = max(Road.roadSet.keys())+1 
  return FILENAME, startIter
  
def options():
  parser = ArgumentParser()
  
  parser.add_argument('json')
  parser.add_argument('png')
  parser.add_argument('--zoom', type=float, default=1.0)
  
  return parser.parse_args()

def distance2Road(pt,road):
  if np.isclose(dist(pt,road.start),0) or np.isclose(dist(pt,road.end),0):
    return 0
  if angle(pt,road.start, road.end) * 2 < np.pi and angle(pt, road.end, road.start)*2 < np.pi:
    return np.sin(angle(pt,road.start, road.end)) * dist(road.start,pt)
  elif angle(pt,road.start, road.end) *2 >= np.pi:
    return dist(road.start, pt)
  else:
    return dist(road.end, pt)

def nearestRoad(pt):
  nearbyRoads = set([r.id for n in Road.tree.getNode(pt).getNeighbors() for r in n.getRoads()])
  neighbs = sorted([(Road.roadSet[r],distance2Road(pt, Road.roadSet[r])) for r in nearbyRoads ], key=lambda x: x[1])  
  if len(neighbs) > 0:
    return neighbs[0]
  else:
    return (None, np.inf)
    
class Zone:
  def __init__(self, road):
    self.pixels = []
    self.road = road
    self.color = road.color()
    
class Passage:
  def __init__(self, road):
    self.pts = []
    self.road = road
    
class Pixel(Node):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.zone = None
    self.road = None

def progressRange(start, end, inc=1, display=False):
  for i in range(start, end, inc):
    progress = int(32*(i - start)/(end-start))
    if display:
      print('[%s%s] %5d' % ('|'*progress, '.'*(32-progress), i), end='\r')
    else:
      print('[%s%s]' % ('|'*progress, '.'*(32-progress)), end='\r')
    yield i
  print('[%s]' % ('|'*32))

def progress(l, display=False):
  for i,n in enumerate(l):
    progress = int(32*i/len(l))
    if display:
      print('[%s%s] %5d' % ('|'*progress, '.'*(32-progress), i), end='\r')
    else:
      print('[%s%s]' % ('|'*progress, '.'*(32-progress)), end='\r')
    yield n
  print('[%s]' % ('|'*32))
    
def main():
  ops = options()
  FILENAME, startIter = load(ops.json, ops.zoom)
  
  
  
  minX = np.inf
  maxX = -np.inf
  minY = np.inf
  maxY = -np.inf
  for n in Node.nodeSet.values():
    minX = min(minX, n.coord[0])
    maxX = max(maxX, n.coord[0])
    minY = min(minY, n.coord[1])
    maxY = max(maxY, n.coord[1])
    
  minX = int(minX - 20)
  maxX = int(maxX + 20)
  minY = int(minY - 20)
  maxY = int(maxY + 20)
    
  qt = Node.tree
  qtDraw(qt, ops.png[:-4] + '_tree.png')
  queue = []
  pixels = {}

  ops.zoom /= 3
  
  skip = []
  checked = []
  orphan = []
  
  
  print('Loading pixels')
  far = None
  ct = 0
  for x in progressRange(minX, maxX, display=True):
    pixels[x] = {}

    y = minY
    while y < maxY:
      trial = Pixel(x,y)
      if far is not None and d - dist(far,trial) > 25 * ops.zoom:
        skip.append(trial)
        y += d - dist(far,trial) - 25 * ops.zoom
        continue
      box = qt.getNode(trial)
      if box.roadCount > 0 and (box.maxX - box.minX)*np.sqrt(2) < 25:
        queue.append(trial)
        pixels[x][y] = trial
        far = None
        y += 1
        continue
      nearbyRoads = set([r.id for n in qt.getNode(trial).getNeighbors() for r in n.getRoads()])
      neighbs = sorted([(Road.roadSet[r],distance2Road(trial, Road.roadSet[r])) for r in nearbyRoads ], key=lambda x: x[1])
      if len(neighbs) > 0 and neighbs[0][1] < 25 * ops.zoom:
        # print('Distance from (%3d,%3d) to %s: %.3f' % (x,y,str(neighbs[0][0]),neighbs[0][1]))
        if neighbs[0][1]*2 < max(1,np.sqrt(neighbs[0][0].level)):
          trial.road = closeRoad[0]
          ct += 1
        else:
          queue.append(px)        
        pixels[x][y] = trial
        far = None
      elif len(neighbs) > 0:
        far = trial
        d = neighbs[0][1]
        checked.append(trial)
      else:
        orphan.append(trial)
        pass
      y += 1
  print('Considering %d pixels' % len(queue))
  random.shuffle(queue)
  print('%d road pixels found' % ct)
  diff = 1
  print('Cleaning roads')
  it = 0
  ct = 0
  while diff > 0:
    queue = secondqueue
    secondqueue = []
    start = len(queue)
    for px in progress(queue):
      # if pixels on either side are part of a road, then this one is too
      x,y = px.coord
      horiz = [pixels[x][Y] for Y in range(y-1,y+2,2) if Y in pixels[x]]
      verts = [pixels[X][y] for X in range(x-1,x+2,2) if X in pixels and y in pixels[X]]
      if sum(p.road is not None for p in horiz) == 2:
        px.road = horiz[0].road
        ct += 1
      elif sum(p.road is not None for p in verts) == 2:
        px.road = verts[0].road
        ct += 1
      else:
        secondqueue.append(px)
    
    diff = start - len(secondqueue)
  print('%d more road pixels found' % ct)
  zones = []
  it = 0
  diff = 1
  print('Finding zones')
  while diff > 0:
    queue = secondqueue
    random.shuffle(queue)
    secondqueue = []
    
    start = len(queue)
    print('%3d: %5d' % (it, len(queue)))
    it += 1
    for px in progress(queue):
      # if this is next to a road, make it a building
      x, y = px.coord
      horiz = [pixels[x][Y] for Y in range(y-1,y+2,2) if Y in pixels[x]]
      verts = [pixels[X][y] for X in range(x-1,x+2,2) if X in pixels and y in pixels[X]]
      neighbors = horiz + verts
      for p in neighbors:
        if p.road is not None:
          z = Zone(p.road)
          zones.append(z)
          z.pixels.append(px)
          px.zone = z
          break
      else:
        # if this is next to a building and
        # 1. close enough to the road
        # 2. closer to that road than any other option
        candidates = []
        for p in neighbors:
          if p.zone is not None and distance2Road(px, p.zone.road) < p.zone.road.buildingSize * ops.zoom * np.cbrt(p.zone.road.level+1):
            candidates.append((p.zone,distance2Road(px, p.zone.road)))
        if len(candidates) > 0:
          candidate = sorted(candidates, key=lambda x: x[1])[0]
          px.zone = candidate[0]
          px.zone.pixels.append(px)
        else:
          secondqueue.append(px)
      
    draw(minX, maxX, minY, maxY, zones, ops.png[:-4] + '_%d.png'%it, secondqueue)
    diff = start - len(secondqueue)

  draw(minX, maxX, minY, maxY, zones, ops.png, off=secondqueue, checked=checked, skip=skip, orphan=orphan)
def draw(minX, maxX, minY, maxY, zones, fname, off=None, checked=None, skip=None, orphan=None):    
  img = Image.new('RGBA', (maxX - minX, maxY - minY), (255,255,255,255))
  draw = ImageDraw.Draw(img)
  for z in zones:
    for px in z.pixels:
      x = px.coord[0] - minX
      y = px.coord[1] - minY
      draw.point([(x,y)],fill=z.color)
  
  for px in off or []:
    x = px.coord[0] - minX
    y = px.coord[1] - minY
    draw.point([(x,y)],fill='grey')

  
  for px in checked or []:
    x = px.coord[0] - minX
    y = px.coord[1] - minY
    draw.point([(x,y)],fill='pink')
  
  for px in skip or []:
    x = px.coord[0] - minX
    y = px.coord[1] - minY
    draw.point([(x,y)],fill='yellow')
  
  for px in orphan or []:
    x = px.coord[0] - minX
    y = px.coord[1] - minY
    draw.point([(x,y)],fill='cyan')
    
  img.save(fname)
  

  
if __name__=='__main__':
  main()