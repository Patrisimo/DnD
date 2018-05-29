var pointCosts = []
pointCosts[0] = 0
pointCosts[4] = -3
pointCosts[6] = -2
pointCosts[8] = 0
pointCosts[10] = 2
pointCosts[12] = 5
pointCosts[14] = 9

var statNames = ['STR','CON','DEX','INT','WIS','CHA']
var lowChosen = false
var highChosen = false



function roll() {
  for (var i=1;i<=6;i++) {
    var num = Math.floor(Math.random()*4)+1;
    var field = document.getElementById(i+'d').value = num;
    update(i);
  }
  checkAll()
  
}

function reset1() {
  for (var i=1;i<=6;i++) {
    document.getElementById(i+'d').value = ''
    document.getElementById(i+'stat').selectedIndex = 0
  }
  checkAll()
}

function checkAll() {
  for (var i=1;i<=6;i++) {
    update(i);
  }
  checkStats(1)
  
  for (var i=0;i<6;i++) {
    document.getElementById('buy' + statNames[i]).disabled = true
  }
  document.getElementById('points').innerHTML = '---'
  
  var total = 0
  for (var j=1;j<=6;j++) {
    fillStat(statNames[j-1]);
    var amt = parseInt(document.getElementById(j+'val').innerHTML)
    if (isNaN(amt)) {
      console.log('Not ready to calculate points yet')
      return;
    } else
      total += amt
  }
  
  
  for (var i=0;i<6;i++) {
    document.getElementById('buy' + statNames[i]).disabled = false
  }
  
  document.getElementById('total').innerHTML = 30 - total
  document.getElementById('points').innerHTML = 30 - total - pointsSpent();
  buy()
}

function update(i) {
  i = parseInt(i)
  var entry = document.getElementById(i+'d').value;
  var rval = parseInt(entry);
  console.log('Die ' + i + ' was updated to ' + rval)
  if (isNaN(rval) || rval < 1 || rval > 4) {
    document.getElementById(i+'d').value = '';
    document.getElementById(((i%6)+1) + 'dmod').innerHTML = '';
  } else {
    document.getElementById(i+'d').value = rval
    if (rval == 1) {
      document.getElementById(((i%6)+1) + 'dmod').innerHTML = '+1'
    } else if (rval == 4) {
      document.getElementById(((i%6)+1) + 'dmod').innerHTML = '-1'
    } else {
      document.getElementById(((i%6)+1) + 'dmod').innerHTML = '+0'
    }
  }
  compute_base(i)
  compute_base((i%6)+1)
  
  
}

function compute_base(i) {
  i = parseInt(i);
  var value = parseInt(document.getElementById(i+'d').value);
  var mod = parseInt(document.getElementById(i+'dmod').innerHTML);
  
  if (!isNaN(value) && !isNaN(mod)) {
    document.getElementById(i+'val').innerHTML = value + mod;
  } else {
    document.getElementById(i+'val').innerHTML = ''
  }
}

function myLookup(key, list) {
  for (var i=0;i<list.length;i++) {
    if (list[i][0] == key)
      return i;
  }
  alert('No match four for ' + key + ' in ' + list)
}

function checkStats(start) {
  console.log('Stat updated: ' + start)
  var statAssignments = [];
  statAssignments['nul'] = 0;
  statAssignments['str'] = 0; 
  statAssignments['dex'] = 0;
  statAssignments['con'] = 0;
  statAssignments['int'] = 0;
  statAssignments['wis'] = 0;
  statAssignments['cha'] = 0;
  var statField;
  var stat;
  var statInd;
  for (var i=0;i<6;i++) {
    statInd = (start + i-1) % 6 + 1
    statField = document.getElementById(statInd+'stat')
    stat = statField.options[statField.selectedIndex].value
    console.log('Checking stat #' + statInd + ', currently ' + stat)
    if (stat == 'nul')
      continue;
    if (statAssignments[stat] > 0) {
      console.log('Stat conflicts with ' + statAssignments[stat])
      statField.selectedIndex = 0;
      clearStat(stat.toUpperCase())
    } else {
      statAssignments[stat] = statInd;
    }
  }
  for (let stat in statAssignments) {
    if (stat == 'nul')
      continue
    if (statAssignments[stat] == 0) {
      clearStat(stat.toUpperCase());
    } else {
      fillStat(stat.toUpperCase())
    }
  }
}

function clearStat(stat) { // clears the entry in the 'base stat' area
  console.log('Clearing ' + stat)
  document.getElementById('base'+stat).innerHTML = '---'
  updateStat(stat)
}

function fillStat(stat) { // computes the entry in the 'base stat' area
  console.log('Filling the entry for ' + stat)
  var i=1;
  for (;i<=6;i++) {
    var statField = document.getElementById(i+'stat');
    var assigned = statField.options[statField.selectedIndex].value
    if (assigned.toUpperCase() == stat)
      break
  }
  if (i<=6) {
    console.log('Moving value from ' + i + 'val to base' + stat)
    var basestat = parseInt(document.getElementById(i+'val').innerHTML)+8
    if (isNaN(basestat))
      document.getElementById('base'+stat).innerHTML = '---'
    else
      document.getElementById('base'+stat).innerHTML = basestat
    updateStat(stat)
  }
}

function updateStat(stat) { // update the value in the 'final stat' area
  var base = document.getElementById('base'+stat).innerHTML
  if (base == '---') {
    document.getElementById(stat).innerHTML = '---';
    return;
  }
  var buyField = document.getElementById('buy'+stat)
  var bought = buyField.options[buyField.selectedIndex].value
  console.log('Computing total: ' + parseInt(base) + ' + ' + parseInt(bought))
  var total = parseInt(base) + parseInt(bought)-8
  if (!isNaN(total)) {
    document.getElementById(stat).innerHTML = total;
  }
}

function buy() { // compute the updated stat value, compute remaining points, disable options that are too expensive or not allowed
  var points = parseInt(document.getElementById('total').innerHTML) - pointsSpent()
  document.getElementById('points').innerHTML = points
  var penalty
  console.log('Points remaining: ' + points)
  for (var i=0;i<6;i++) { // disable option that are too expensive
    
    // first, find how many points are being spent here
    field = document.getElementById('buy'+statNames[i])
    var sunkCost = pointCosts[field.options[field.selectedIndex].value]
    if (i == 0 && (field.selectedIndex == 1 || field.selectedIndex == 2))
      sunkCost++;
    
    for (var j=0;j<field.length;j++) { // for each point value
      op = field.options[j]
      if (op.selected)
        continue
      else {
        op.disabled = false
        if (op.value == '4' && lowChosen)
          op.disabled = true
        if (op.value == '14' && highChosen)
          op.disabled = true
        if (i == 0 && (op.value == 4 || op.value == 6))
          penalty = 1;
        else
          penalty = 0;
        console.log('For ' + statNames[i] + ', option ' + op.value + ' costs ' + (pointCosts[op.value] + penalty - sunkCost) + ', currently have ' + points)
        if (pointCosts[op.value] + penalty - sunkCost > points)
          op.disabled = true
        
      }
      
    }
    updateStat(statNames[i])
  }
  
}

function reset2() {
  for (var i=0;i<6;i++) {
    document.getElementById('buy' + statNames[i]).selectedIndex = 2
  }
  checkAll()
}


function pointsSpent() {
  var points = 0;
  lowChosen = false
  highChosen = false
  for (var i=0;i<6;i++) {
    var ptField = document.getElementById('buy' + statNames[i])
    var selected = ptField.options[ptField.selectedIndex].value
    points += pointCosts[selected]
    if (i==0 && pointCosts[selected] < 0)
      points++
    if (selected == '4')
      lowChosen = true;
    if (selected == '14')
      highChosen = true;
    
  }
  return points
}


window.onload = function () {
    for (var i=1;i<=6;i++) {
      document.getElementById('buy'+statNames[i-1]).selectedIndex = 2
      document.getElementById(i+'d').addEventListener('input', checkAll)
      document.getElementById(i+'stat').selectedIndex = i
    }
    checkAll()
    
}