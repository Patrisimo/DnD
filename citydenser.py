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
from buildings import nearestRoad, progress
from citybuilder3 import makeRoad, closeNode, load, save, cycle, drawCity
import traceback

  
def options():
  parser = ArgumentParser()
  parser.add_argument('filename')
  parser.add_argument('output')
  return parser.parse_args()
  
  
def main():
  # Idea is to, for each box, add a node somewhere in that box
  ops = options()
  
  FILENAME, startIter = load(ops.filename)
  
  qt = Node.tree
  boxes = [qt.root]
  i=0
  while i<len(boxes):
    boxes.extend(boxes[i].children)
    i+=1
  ct = 0
  Timer.start('Filling boxes')
  for b in progress(boxes):
    # 20 attempts
    attempts=0
    d = 0
    while attempts < 20 and (d < Road.minLength or d > 2*Road.minLength):
      x = (b.maxX - b.minX)*random.random() + b.minX
      y = (b.maxY - b.minY) * random.random() + b.minY
      n = Node(x,y)
      road, d = nearestRoad(n)
      attempts += 1
    if attempts < 20:
      n.add()
      (n1, n2), _, _ = closeNode(n, Node.nodeSet, Road.roadSet)
      roadType = type(road)
      if roadType is SpecialRoad:
        roadType = ResidentialRoad 
      makeRoad(n1, n2, Road.roadSet, roadType)
      ct += 1
  
  Timer.stop('Filling boxes')
  print('Added %d nodes' % ct)
  try:
    cycle()
  except:
    print('Stopped')
    traceback.print_exc()
  print('Finished')
  drawCity('%s.png' % ops.output, clean=True, clusters=False)
  save('%s.json' % ops.output, startIter)
  print(Timer.report())

if __name__=='__main__':
  main() 