import time

class Timer:
  startTimes = {}
  
  elapsedTimes = {}
  
  def start(label):
    assert Timer.startTimes.get(label, 0) == 0
    Timer.startTimes[label] = time.time()
  
  def stop(label):
    assert Timer.startTimes.get(label, 0) > 0
    total = time.time() - Timer.startTimes[label]
    Timer.startTimes[label] = 0
    Timer.elapsedTimes[label] = Timer.elapsedTimes.get(label, 0) + total
    
  def report():
    s = ''
    for label, amt in Timer.elapsedTimes.items():
      s += '%s: %f\n' % (label, amt)
    return s