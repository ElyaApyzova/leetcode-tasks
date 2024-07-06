//Input: (intervals = [
//[1, 3],
//[6, 9],
//]),
//(newInterval = [2, 5]);
//Output: [
//[1, 5],
//[6, 9],
//];

const insert = function (intervals, newInterval) {
  let n = intervals.length,
    i = 0;
  res = [];

  while (i < n && intervals[i][1] < newInterval[0]) {
    res.push(intervals[i]);
    i++;
  }

  while (i < n && newInterval[1] >= intervals[i][0]) {
    newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
    newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
    i++;
  }
  res.push(newInterval);

  while (i < n) {
    res.push(intervals[i]);
    i++;
  }
  return res;
};

//Input: s = "Hello World"
//Output: 5
//Explanation: The last word is "World" with length 5.

const lengthOfLastWord = function (s) {
  let p = s.length - 1;
  while (p >= 0 && s[p] === " ") {
    p--;
  }

  let length = 0;
  while (p >= 0 && s[p] !== " ") {
    p--;
    length++;
  }
  return length;
};


// Input: n = 3
//Output: [[1,2,3],[8,9,4],[7,6,5]]

const generateMatrix = function (n) {
  const result = new Array(n).fill(0).map(() => new Array(n).fill(0));
  const dirs = [
    [0, 1],
    [1, 0],
    [0, -1]
    [-1, 0],
  ];
  let d = 0;
  let row = 0;
  let col = 0;
  let cnt = 1;

  while (cnt <= n * n) {
   result[row][col] = cnt++;
   let newRow = (row + (dirs[d][0] % n) + n) % n;
   let newCol = (col + (dirs[d][1] % n) + n) % n;
   if (result[newRow][newCol] != 0) d = (d + 1) % 4;

   row += dirs[d][0]
   col += dirs[d][1]
  }
  return result;
};

//Input: n = 3, k = 3
//Output: "213"

const getPermutation = function (n, k) {
  let factorials = new Array(n);
  let nums = ["1"];
  factorials[0] = 1;
  for (let i = 1; i < n; ++i) {
    factorials[i] = factorials[i - 1] * i;
    nums.push((i + 1).toString());
  }
  --k;
  let output = "";
  for (let i = n - 1; i > -1; --i) {
    let idx = Math.floor(k / factorials[i]);
    k -= idx * factorials[i];
    output += nums[idx];
    nums.splice(idx, 1);
  }
  return output;
};


//Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
//Output: [[1,6],[8,10],[15,18]]
//Explanation: Since intervals [1,3] and [2,6] overlap, merge them into 
//[1,6].

const overlap = function (a, b) {
  return a[0] <= b[1] && b[0] <= a[1];
};

const buildGraph = function (intervals) {
  const graph = new Map();
  for (const i = 0; i < intervals.length; i++) {
    for (const j = i + 1; j < intervals.length; j++) {
      if (overlap(intervals[i], intervals[j])) {
        if (graph.has(intervals[i])) {
          graph.get(intervals[i]).push(intervals[j]);


        } else {
          graph.set(intervals[i], [intervals[j]]);
        }
        if (graph.has(intervals[j])) {
          graph.get(intervals[j]).push(intervals[i]);
        
        } else {
          graph.set(intervals[j], [intervals[i]]);
        }

      }
    }
  }

  return graph;
};

const mergeNodes = function (nodes) {
  const minStart = Infinity;
  const maxEnd = -Infinity;
  for (let node of nodes) {
    minStart = Math.min(minStart, node[0]);
    maxEnd = Math.max(maxEnd, node[1]);
  }
  return [minStart, maxEnd];
};


