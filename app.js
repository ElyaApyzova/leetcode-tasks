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


const markComponentDFS = function (
  start,
  graph,
  nodesInComp,
  compNumber,
  visited,
) {
  const stack = [start];
  while (stack.length) {
    const node = stack.pop();
    if (!visited.has(node)) {
      visited.add(node);
      if (nodesInComp[compNumber]) {
        nodesInComp[compNumber].push(node)
      } else {
        nodesInComp[compNumber] = [node];
      }
      if (graph.has(node)) {
        for (let child of graph.get(node)) {
          stack.push(child);
        }
      }
    }
  }
};

const merge = function (intervals) {
  const graph = buildGraph(intervals);
  const nodesInComp = {};
  const visited = new Set();
  const compNumber = 0;
  for (let interval of intervals) {
    if (!visited.has(interval)) {
      markComponentDFS(interval, graph, nodesInComp, compNumber, visited);
      compNumber++;
    }
  }
  var merged = [];
  for (var comp = 0; comp < compNumber; comp++) {
    merged.push(mergeNodes(nodesInComp[comp]));
  }
  return merged;
};



//Input: head = [0,1,2], k = 4
//Output: [2,0,1]


const rotateRight = function (head, k) {
  if (head == null) return null;
  if (head.next == null) return head;

  let old_tail = head;
  let n;
  for (n = 1; old_tail.next != null; n++) old_tail = old_tail.next;
  old_tail.next = head;

  let new_tail = head;
  for (let i = 0; i < n - (k % n) -1; i++) new_tail = new_tail.next;
  let new_head = new_tail.next;

  new_tail.next = null;
  return new_head;

};


//Input: m = 3, n = 7
//Output: 28

const uniquePaths = function (m,n) {
  if (m == 1 || n == 1) {
    return 1;
  }
  return uniquePaths(m - 1, n) + uniquePaths(m, n - 1)
};


//Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
//Output: 2
//Explanation: There is one obstacle in the middle of the 3x3 grid above.
//There are two ways to reach the bottom-right corner:
//1. Right -> Right -> Down -> Down
//2. Down -> Down -> Right -> Right


const uniquePathsWithObstacles = function (obstacleGrid) {
  let R = obstacleGrid.length;
  let C = obstacleGrid[0].length;
  if (obstacleGrid[0][0] == 1) {
    return 0;
  }

  obstacleGrid[0][0] = 1;
  for (let i = 1; i < R; i++) {
    obstacleGrid[i][0] = obstacleGrid[i][0] == 0 && obstacleGrid[i - 1][0] == 1 ? 1 : 0;
  }
  for (let i = 1; i < C; i++) {
    obstacleGrid[0][i] = obstacleGrid[0][i] == 0 && obstacleGrid[0][i - 1] == 1 ? 1 : 0;
  }
  for (let i = 1; i < R; i++) {
    for (let j = 1; j < C; j++) {
      if (obstacleGrid[i][j] == 0) {
        obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1];
      } else {
        obstacleGrid[i][j] = 0;
      }
    }
  }
  return obstacleGrid[R - 1][C - 1]
};

//https://leetcode.com/problems/minimum-path-sum/

//Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
//Output: 7
//Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.


const minPathSum = function (grid) {
  let dp = new Array(grid.length)
   .fill()
   .map(() => new Array(grid[0].length).fill(0));
   for (let i = grid.length - 1; i >= 0; i--) {
    for (let j = grid[0].length - 1; j >= 0; j--) {
      if (i === grid.lenth - 1 && j !== grid[0].length - 1)
        dp[i][j] = grid[i][j] + dp[i][j + 1];
      else if (j === grid[0].length - 1 && i !== grid.length - 1 );

      dp[i][j] = grid[i][j] + dp[i + 1][j];
       if (j !== grid[0].length - 1 && i !== grid.length - 1)
        dp[i][j] = grid[i][j] + Math.min(dp[i + 1][j], dp[i][j + 1]);
      else dp[i][j] = grid[i][i][j]
    }
   }
   return dp[0][0];
}


// https://leetcode.com/problems/valid-number/

//Input: s = "0"

//Output: true


const isNumber = function (s) {
  const seenDigit = false;
  const seenExponent = false;
  const seenDot = false;
  for (let i = 0; i < s.length; i++) {
    let curr = s[i];
    if (!isNaN(curr)) {
      seenDigit = true;
    } else if (curr == "+" || curr == "-") {
      if (i > 0 && s[i - 1] != "e" && s[i - 1] != "E") {
        return false;
      }
    } else if (curr == "e" || curr == "E") {
      if (seenExponent || !seenDigit) {
        return false;
      }
      seenExponent = true;
      seenDigit = false;
    } else if (curr == ".") {
      if (seenDot || seenExponent) {
        return false;
      }
      seenDot = true;
    } else {
      return false;
    }
  }
  return seenDigit;
};




//https://leetcode.com/problems/plus-one/description/


//Input: digits = [1,2,3]
//Output: [1,2,4]
//Explanation: The array represents the integer 123.
//Incrementing by one gives 123 + 1 = 124.
//Thus, the result should be [1,2,4].

const plusOne = function (digits) {
  let n = digits.length;
  for (let i = n - 1; i >= 0; --i) {
    if (digits[i] == 9) {
      digits[i] = 0;
    } else {
      digits[i]++;
      return digits;
  }
  }
  digits.unshift(1);
  return digits;
};


// https://leetcode.com/problems/add-binary/description/


//Input: a = "11", b = "1"
//Output: "100"


const addBinary = function (a, b) {
  let n = a.length,
      m = b.length;
      if (n < m) return addBinary(b, a);

      let result = [];
      let carry = 0,
        j = m - 1;
        for (let i = n - 1; i >= 0; --i) {
          if (a[i] === "1") ++carry;
          if (j >= 0 && b[j--] === "1") ++carry;

          result.push((carry % 2).toString());
          carry = Math.floor(carry / 2);
        }
        if (carry === 1) result.push("1");
        return result.reverse().join("");
};


//https://leetcode.com/problems/text-justification/description/


//Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
//Output:
//[
   //"This    is    an",
   //"example  of text",
   //"justification.  "
//]


const fullJustify = function (words, maxWidth) {
  let ans = [];
  let i = 0;
  while (i < words.length) {
    let currentLine = getWords(i, words, maxWidth);
    i += currentLine.length;
    ans.push(createLine(currentLine, i, words, maxWidth));
  }
  return ans;
}