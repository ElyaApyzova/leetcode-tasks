//Input: (intervals = [
//[1, 3],
//[6, 9],
//]),
//(newInterval = [2, 5]);
//Output: [
//[1, 5],
//[6, 9],
//];

const { TreeNode } = require("antd/es/tree-select");

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

  function getWords(i, words, maxWidth) {
    let currentLine = [];
    let currLength = 0;
    while (i < words.length && currLength + words[i].length <= maxWidth) {
      currentLine.push(words[i]);
      currLength += words[i].length + 1;
      i++;
    }
    return currentLine;
  }

  function createLine(line, i, words, maxWidth) {
    let baseLength = -1;
    for (let word of line) {
      baseLength += word.length + 1;
    }
    let extraSpaces = maxWidth - baseLength;
    if (line.length === 1 || i === words.length) {
      return line.join(" ") + " ".repeat(extraSpaces);
    }
    let wordCount = line.length - 1;
    let spacesPerWord = Math.floor(extraSpaces / wordCount);

    let needExtraSpace = extraSpaces % wordCount;
    for (let j = 0; j < needsExtraSpace; j++) {
      line[j] += " ";
    }
    for (let j = 0; j < wordCount; j++) {
      line[j] += " ".repeat(spacesPerWord);
    }
    return line.join(" ");
  }
};




//https://leetcode.com/problems/sqrtx/description/

//Input: x = 4
//Output: 2
//Explanation: The square root of 4 is 2, so we return 2.


const mySqrt = function (x) {
  if (x < 2) return x;
  let num;
  let pivot,
      left = 2,
      right = Math.floor(x / 2);
      while (left <= right) {
        pivot = left + Math.floor((right - left) / 2);
        num = pivot * pivot;
        if (num > x) right = pivot - 1;
        else if (num < x) left = pivot + 1;
        else return pivot;
      }
      return right;
}

//https://leetcode.com/problems/climbing-stairs/description/


//Input: n = 2
//Output: 2
//Explanation: There are two ways to climb to the top.
//1. 1 step + 1 step
//2. 2 steps

const climbStairs = function (n) {
  return climb_Stairs(0, n);
};

const climb_Stairs = function (i, n) {
  if (i > n) {
    return 0;
  }
  if (i == n) {
    return 1
  }
  return climb_Stairs(i + 1, n) + climb_Stairs(i + 2, n);
};


//https://leetcode.com/problems/simplify-path/description/


//Input: path = "/home/"

//Output: "/home"

//Explanation:

//The trailing slash should be removed.

const simplifyPath = function (path) {
  let stack = [];
  for (let portion of path.split("/")) {
    if (portion === "..") {
      if (stack.length) {
        stack.pop();
      }
    } else if (portion !== "." && portion) {
      stack.push(portion);
    }
  }
  return "/" + stack.join("/");
};

//https://leetcode.com/problems/edit-distance/

//Input: word1 = "horse", word2 = "ros"
//Output: 3
//Explanation: 
//horse -> rorse (replace 'h' with 'r')
//rorse -> rose (remove 'r')
//rose -> ros (remove 'e')


const minDistance = function (word1, word2) {
  let memo = Array(word1.length + 1)
      .fill()
      .map(() => Array(word2.length + 1).fill(null));
      function minDistanceRecur(word1, word2, word1Index, word2Index) {
        if (word1Index === 0) {
          return word2Index;
        }
        if (word2Index === 0) {
          return word1Index;
        }
        if (memo[word1Index][word2Index] !== null) {
          return memo[word1Index][word2Index];
        }
        let minEditDistance = 0;
        if (word1[word1Index -1] === word2[word2Index - 1]) {
          minEditDistance = minDistanceRecur(
            word1,
            word2,
            word1Index - 1,
            word2Index - 1,
          );
        } else {
          let insertOperation = minDistanceRecur(
           word1,
           word2,
           word1Index,
           word2Index - 1,
          );
          let deleteOperation = minDistanceRecur (
            word1,
            word2,
            word1Index - 1,
            word2Index,
          );
          let replaceOperation = minDistanceRecur(
            word1,
            word2,
            word1Index - 1,
            word2Index - 1,
          );
          minEditDistance = Math.min(insertOperation, Math.min(deleteOperation, replaceOperation),) + 1;
        }
        memo[word1Index][word2Index] = minEditDistance;
        return minEditDistance;
      }
      return minDistanceRecur(word1, word2, word1.length, word2.length);
};


//https://leetcode.com/problems/set-matrix-zeroes/

//Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
//Output: [[1,0,1],[0,0,0],[1,0,1]]

const setZeroes = function (matrix) {
  let isCol = false;
  let R = matrix.length;
  let C = matrix[0].length;
  for (let i = 0; i < R; i++) {
    if (matrix[i][0] == 0) {
      isCol = true;
    }
    for (let j = 1; j < C; j++) {
      if (matrix[i][j] == 0) {
        matrix[0][j] = 0;
        matrix[i][0] = 0;
      }
    }
  }
  for (let i = 1; i < R; i++) {
    for (let j = 1; j < C; j++) {
      if (matrix[i][0] == 0 || matrix[0][j] == 0) {
        matrix[i][j] = 0;
      }
    }
  }
  if (matrix[0][0] == 0) {
    for (let j = 0; j < C; j++) {
      matrix[0][j] = 0;
    }
  }
  if (isCol) {
    for (let i = 0; i < R; i++) {
      matrix[i][0] = 0;
    }
  }
};



//https://leetcode.com/problems/search-a-2d-matrix/


//Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
//Output: true


const searchMatrix = function (matrix, target) {
  let m = matrix.length;
  if (m == 0) return false;
  let n = matrix[0].length;
  let left = 0,
  right = m * n - 1;
  let pivotIdx, pivotElement;
  while (left <= right) {
    pivotIdx = Math.floor((left + right) / 2);
    pivotElement = matrix[Math.floor(pivotIdx / n)][pivotIdx % n];
    if (target == pivotElement) return true;
   else {
    if (target < pivotElement) right = pivotIdx - 1;
    else left = pivotIdx + 1;
  }
}
  return false;
};



//https://leetcode.com/problems/sort-colors/

//Input: nums = [2,0,2,1,1,0]
//Output: [0,0,1,1,2,2]

const sortColors = function (nums) {
  let p0 = 0,
  curr = 0;
  let p2 = nums.length - 1;
  while (curr <= p2) {
    if (nums[curr] == 0) {
      [nums[curr++], nums[p0++]] = [nums[p0], nums[curr]];
    } else if (nums[curr] == 2) {
      [nums[curr], nums[p2--]] = [nums[p2], nums[curr]];
    } else curr++;
  }
};



//https://leetcode.com/problems/minimum-window-substring/description/

//Input: s = "ADOBECODEBANC", t = "ABC"
//Output: "BANC"
//Explanation: The minimum window substring "BANC" includes 'A', 'B', and //'C' from string t.



const  minWindow = function (s, t) {
  if (s.length === 0 || t.length === 0) {
    return "";
  }

  let dictT = new Map();
  for (let i = 0; i < t.length; i++) {
    let count = dictT.get(t.charAt(i)) || 0;
    dictT.set(t.charAt(i), count + 1);
  }

  let required = dictT.size;
  let l = 0,
  r = 0;
  let formed = 0;
  let windowCounts = new Map();
  let ans = [-1, 0, 0];
  while (r < s.length) {
    let c = s.charAt(r);
    let count = windowCounts.get(c) || 0;
    windowCounts.set(c, count + 1);
    if (dictT.has(c) && windowCounts.get(c) === dictT.get(c)) {
      formed++;
    }
    while (l <= r && formed === required) {
      c = s.charAt(l);
      if (ans[0] === -1 || r - l + 1 < ans[0]) {
        ans[0] = r - l + 1;
        ans[1] = l;
        ans[2] = r;
      }
      windowCounts.set(c, windowCounts.get(c) - 1);
      if (dictT.has(c) && windowCounts.get(c) < dictT.get(c)) {
      
      formed--;
    }
    l++;
  }
  r++;
}
return ans[0] === -1 ? "" : s.substring(ans[1], ans[2] + 1);
}



//https://leetcode.com/problems/combinations/

//Input: n = 4, k = 2
//Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
//Explanation: There are 4 choose 2 = 6 total combinations.
//Note that combinations are unordered, i.e., [1,2] and [2,1] are considered //to be the same combination.



const combine = function (n, k) {
  const ans = [];
  const backtrack = (curr, firstNum) => {
    if (curr.length === k) {
      ans.push([...curr]);
      return;
    }

    const need = k - curr.length;
    const remain = n - firstNum + 1;
    const available = remain - need;
    for (let num = firstNum; num <= firstNum + available; num++) {
      curr.push(num);
      backtrack(curr, num + 1);
      curr.pop();
    }
  };
  backtrack([], 1);
  return ans;
};



//https://leetcode.com/problems/subsets/description/



//Input: nums = [1,2,3]
//Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]



const subsets = function (nums) {
  let output = [];
  let n = nums.length;
  function backtrack(first = 0, curr = [], k) {
    if (curr.length == k) {
      output.push([...curr]);
      return;
    }
    for (let i = first; i < n; i++) {
      curr.push(nums[i]);
      backtrack(i + 1, curr, k);
      curr.pop();
    }
  }

  for (let k = 0; k < n + 1; k++) {
    backtrack(0, [], k);
  }
  return output;
};




//79  https://leetcode.com/problems/word-search/description/


//Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], //word = "ABCCED"
//Output: true


const exist = function (board, word) {
  const ROWS = board.length;
  const COLS = board[0].length;
  const backtrack = function (row, col, suffix) {
    if (suffix.length == 0) return true;
    if (
      row < 0 ||
      row == ROWS ||
      col < 0 ||
      col == COLS ||
      board[row][col] != suffix.charAt(0)
    )
   return false;
    let ret = false;
    board[row][col] = "#" 
    const directions = [
      [0, 1],
      [1, 0],
      [0, -1],
      [-1, 0]
    ];
    
    for (let [rowOffset, colOffset] of directions) {
      ret = backtrack(row + rowOffset, col + colOffset, suffix.slice(1));
      if (ret) break;
    }
    board[row][col] = suffix.charAt(0);
    return ret;
  };

  for (let row = 0; row < ROWS; ++row) {
    for (let col = 0; col < COLS; ++col) {
      if (backtrack(row, col, word)) return true;
    }
  }
  return false;
};



//80 https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/


//Input: nums = [1,1,1,2,2,3]
//Output: 5, nums = [1,1,2,2,3,_]
//Explanation: Your function should return k = 5, with the first five //elements of nums being 1, 1, 2, 2 and 3 respectively.
//It does not matter what you leave beyond the returned k (hence they are //underscores).


const removeDuplicates = function (nums) {
  let j = 0;
  for (let i = 0; i < nums.length; i++) {
    if (j < 2 || nums[i] > nums[j - 2]) {
      nums[j++] = nums[i];
    }
  }
  return j;
};


//81  https://leetcode.com/problems/search-in-rotated-sorted-array-ii/

//Input: nums = [2,5,6,0,0,1,2], target = 0
//Output: true


const search = function (nums, target) {
  let n = nums.length;
  if (n == 0) return false;
  let end = n - 1;
  let start = 0;
  while (start <= end) {
    let mid = start + Math.floor((end - start) / 2);
    if (nums[mid] == target) {
      return true;
    }
    if (!isBinarySearchHelpful(nums, start, nums[mid] )) {
      
      start++;
      continue;
    }
    let pivotArray = existsInFirst(nums, start, nums[mid]);
    let targetArray = existsInFirst(nums, start, target);

    if (pivotArray ^ targetArray) {
      if (pivotArray) {
        start = mid + 1;
      } else {
        end = mid - 1;
      }
    }
  }
  return false;
};

function isBinarySearchHelpful(nums, start, element) {
  return nums[start] != element;
}

function existsInFirst(nums, start, element) {
  return nums[start] <= element;
}


//82   https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/description/


//Input: head = [1,2,3,3,4,4,5]
//Output: [1,2,5]


const deleteDuplicates = function (head) {
  let sentinel = new ListNode(0, head);
  let pred = sentinel;
  while (head !== null) {
    if (head.next !== null && head.val === head.next.val) {
      while (head.next !== null && head.val === head.next.val) {
        head = head.next;
      }
      pred.next = head.next;
    } else {
      pred = pred.next;
    }
    head = head.next;
  }
  return sentinel.next;
};



//83   https://leetcode.com/problems/remove-duplicates-from-sorted-list/


// Input: head = [1,1,2]
//Output: [1,2]



const delDuplicates = function (head) {
  let current = head;
  while (current !== null && current.next !== null) {
    if (current.next.val === current.val) {
      current.next = current.next.next;
    } else {
      current = current.next;
    }
  }
  return head;
};


//84    https://leetcode.com/problems/largest-rectangle-in-histogram/


//Input: heights = [2,1,5,6,2,3]
//Output: 10
//Explanation: The above is a histogram where width of each bar is 1.
//The largest rectangle is shown in the red area, which has an area = 10 //units.



const largestRectangleArea = function (heights) {
  let max_area = 0;
  for (let i = 0; i < heights.length; i++) {
    for (let j = i; j < heights.length; j++) {
      let min_height = Infinity;
      for (let k = i; k <= j; k++) {
        min_height = Math.min(min_height, heights[k]);
      }
      max_area = Math.max(max_area, min_height * (j - i + 1));
    }
  }
  return max_area;
};


//85  https://leetcode.com/problems/maximal-rectangle/


//Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1",//"1","1"],["1","0","0","1","0"]]
//Output: 6
//Explanation: The maximal rectangle is shown in the above picture.


const maximalRectangle = function (matrix) {
  if (matrix.length === 0)  return 0;
  let maxarea = 0;
  let dp = Array(matrix.length)
    .fill(0)
    .map(() => Array(matrix[0].length).fill(0));
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[0].length; j++) {
        if (matrix[i][j] === "1") {
          dp[i][j] = j === 0 ? 1 : dp[i][j - 1] + 1;
          let width = dp[i][j];
          for (let k = i; k >= 0; k--) {
            width = Math.min(width, dp[k][j]);
            maxarea = Math.max(maxarea, width * (i - k + 1));
          }
        }
      }
    }
    return maxarea;
};


// 86   https://leetcode.com/problems/partition-list/



//Input: head = [1,4,3,2,5,2], x = 3
//Output: [1,2,2,4,3,5]


const partition = function (head, x) {
  let before_head = new ListNode(0);
  let before = before_head;
  let after_head = new ListNode(0);
  let after = after_head;
  while (head != null) {
    if (head.val < x) {
      before.next = head;
      before = before.next;
    } else {
      after.next = head;
      after = after.next;
    }
    head = head.next;
  }
  after.next = null;
  before.next = after_head.next;
  return before_head.next;
};


// 87   https://leetcode.com/problems/scramble-string/description/


//Input: s1 = "great", s2 = "rgeat"
//Output: true
//Explanation: One possible scenario applied on s1 is:
//"great" --> "gr/eat" // divide at random index.
//"gr/eat" --> "gr/eat" // random decision is not to swap the two //substrings and keep them in order.
//"gr/eat" --> "g/r / e/at" // apply the same algorithm recursively on both //substrings. divide at random index each of them.
//"g/r / e/at" --> "r/g / e/at" // random decision was to swap the first //substring and to keep the second substring in the same order.
//"r/g / e/at" --> "r/g / e/ a/t" // again apply the algorithm recursively, //divide "at" to "a/t".
//"r/g / e/ a/t" --> "r/g / e/ a/t" // random decision is to keep both //substrings in the same order.
//The algorithm stops now, and the result string is "rgeat" which is s2.
//As one possible scenario led s1 to be scrambled to s2, we return true.




const isScramble = function (s1, s2) {
   const n = s1.length;
   let dp = new Array(n + 1)
      .fill(0)
      .map(() => new Array(n).fill(0).map(() => new Array(n).fill(false)));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          dp[1][i][j] = s1.chart(i) == s2.charAt(j);
        }
      }
      for (let length = 2; length <= n; length++) {
        for (let i = 0; i < n + 1 - length; i++) {
          for (let j = 0; j < n +1 - length; j++) {
            for (let newLength = 1; newLength < l; newLength++) {

              const dp1 = dp[newLength][i];
              const dp2 = dp[length - newLength][i + newLength];

              dp[length][i][j] |= dp1[j] && dp2[j + newLength];
              dp[length][i][j] |= dp1[j + length - newLength] && dp2[j]
            }
          }
        }
      }
      return dp[n][0][0];
};


//88   https://leetcode.com/problems/merge-sorted-array/


//Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
//Output: [1,2,2,3,5,6]
//Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
//The result of the merge is [1,2,2,3,5,6] with the underlined elements //coming from nums1.


const mergeSorted = function (nums1, m, nums2, n) {
  let nums1Copy = nums1.slice(0, m);
  let p1 = 0;
  let p2 = 0;
  for (let p = 0; p < m + n; p++) {
    if (p2 >= n || (p1 < m && nums1Copy[p1] < nums2[p2])) {
      nums1[p] = nums1Copy[p1++];
    } else {
      nums1[p] = nums2[p2++]
    }
  }
};


//89  https://leetcode.com/problems/gray-code/description/


//Input: n = 2
//Output: [0,1,3,2]
//Explanation:
//The binary representation of [0,1,3,2] is [00,01,11,10].
//- 00 and 01 differ by one bit
//- 01 and 11 differ by one bit
//- 11 and 10 differ by one bit
//- 10 and 00 differ by one bit
//[0,2,3,1] is also a valid gray code sequence, whose binary representation //is [00,10,11,01].
//- 00 and 10 differ by one bit
//- 10 and 11 differ by one bit
//- 11 and 01 differ by one bit
//- 01 and 00 differ by one bit



const grayCode = function (n) {
  const res = [0]; 
  const seen = new Set(res);
  const helper = (n, res, seen) => {
    if (res.length === Math.pow(2, n)) {
      return true;
    }
    const curr = res[res.length - 1];
    for (let i = 0; i < n; i++) {
      const next = curr ^ (1 << i);
      if (!seen.has(next)) {
        seen.add(next);
        res.push(next);
        if (helper(n, res, seen)) return true;
        seen.delete(next);
        res.pop();
      }
    }
    return false;
  };
  helper(n, res, seen);
  return res;
};


//90  https://leetcode.com/problems/subsets-ii/


//Input: nums = [1,2,2]
//Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]


const subsetsWithDup = function ( nums) {
  let n = nums.length;
  nums.sort();
  let subsets = [];
  let seen = new Set();
  let maxNumberOfSubsets = Math.pow(2, n);
  for (let subsetIndex = 0; subsetIndex < maxNumberOfSubsets; subsetIndex++) {
    let currentSubset = [];
    let hashcode = "";
    for (let j = 0; j < n; j++) {
      let mask = 1 << j;
      let isSet = mask & subsetIndex;
      if (isSet != 0) {
        currentSubset.push(nums[j]);
        hashcode += nums[j] + ",";
      }
    }
    if (!seen.has(hashcode)) {
      subsets.push(currentSubset);
      seen.add(hashcode);
    }
  }
  return subsets;
};


// 91   https://leetcode.com/problems/decode-ways/description/


//Input: s = "12"
//Output: 2
//Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).


const numDecodings = function (s) {
  let memo = new Object();
  return recursiveWithMemo(0, s, memo);
};

const recursiveWithMemo = (index, str, memo) => {
  if (memo.hasOwnProperty(index)) {
    return memo[index];
  }

  if (index == str.length) {
    return 1;
  } 
  
  if (str.charAt(index) == "0") {
    return 0;
  }
  if (index == str.length - 1) {
    return 1;
  }

  let ans = recursiveWithMemo(index + 1, str, memo);
  if (parseInt(str.substring(index, index + 2)) <= 26) {
    ans += recursiveWithMemo(index + 2, str, memo);
  }
  memo[index] = ans;
  return ans;
};


//  92   https://leetcode.com/problems/reverse-linked-list-ii/description/

//Input: head = [1,2,3,4,5], left = 2, right = 4
//Output: [1,4,3,2,5]


const reverseBetween = function (head, m, n) {
  let left = head,
    stop = false;
    const recurseAndReverse = (right, m, n) => {
      if (n == 1) return;
      right = right.next;
      if (m > 1) left = left.next;
      recurseAndReverse(right, m - 1, n - 1);
      if (left == right || right.next == left) stop = true;

      if (!stop) {
        [left.val, right.val] = [right.val, left.val];
        left = left.next;
      }
    };
    recurseAndReverse(head, m, n)
    return head;
};


//93  https://leetcode.com/problems/restore-ip-addresses/description/


//Input: s = "12"
//Output: 2
//Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).


const restoreIpAddresses = function (s) {
  let ans = [];
  function valid(s, start, length) {
    return (
      length == 1 || (s.charAt(start) != "0" && (length < 3 || s.substring(start, start + length) <= 255))
    );
  }

  function helper(s, startIndex, dots, ans) {
    let remainingLength = s.length - startIndex;
    let remainingNumberOfIntegers = 4 - dots.length;

    if (
      remainingLength > remainingNumberOfIntegers * 3 || remainingLength < remainingNumberOfIntegers
    ) {
      return;
    }
    if (dots.length == 3) {
      if (valid(s, startIndex, remainingLength)) {
        let temp = "";
        let last = 0;
        for (let dot of dots) {
          temp += s.substring(last, last + dot) + ".";
          last += dot;
        }
        temp += s.substring(startIndex);
        ans.push(temp);
      }
      return;
    }
    for (
      let curPos = 1;
      curPos <= 3 && curPos <= remainingLength;
      ++curPos

    ) {
      dots.push(curPos);
      if (valid(s, startIndex, curPos)) {
        helper(s, startIndex + curPos, dots, ans);
      }
      dots.pop();
    }
  }
 helper(s, 0, [], ans);
 return ans;
};


//https://leetcode.com/problems/binary-tree-inorder-traversal/description/


//Input: root = [1,null,2,3]
//Output: [1,3,2]


const inorderTraversal = function (root) {
  let res = [];
  helper(root, res);
  returnres;
};

const helper = function (root, res) {
  if (root !== null) {
    helper(root.left, res);
    res.push(root.val);
    helper(root.right, res);
  }
};

//95 https://leetcode.com/problems/unique-binary-search-trees-ii/

//Input: n = 3
//Output: [[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,//null,1]]


const allPossibleBST = function (start, end, memo) {
  let res = [];
  if (start > end) {
    res.push(null);
    return res;
  }

  let key = start + "," + end;
  if (memo[key] != undefined) {
    return memo[key];
  }
  for (let i = start; i <= end; ++i) {
    let leftSubTrees = allPossibleBST(start, i - 1, memo);

    let rightSubTrees = allPossibleBST(i + 1, end, memo);

    for (let j = 0; j < leftSubTrees.length; j++) {
      for (let k = 0; k < rightSubTrees.length; k++) {
        let root = new TreeNode(i, leftSubTrees[j], rightSubTrees[k]);
        res.push(root);
      }
    }
  }
 memo[key] = res;
 return res;
};
const generateTrees = function (n) {
  let memo = {};
  return allPossibleBST(1, n, memo);
};


//96 https://leetcode.com/problems/unique-binary-search-trees/description/

//Input: n = 3
//Output: 5


const numTrees = function (n) {
  let G = new Array(n + 1).fill(0);
  G[0] = 1;
  G[1] = 1;
  for (let i = 2; i <= n; i++) {
    for (let j = 1; j <= i; j++) {
      G[i] += G[j - 1] * G[i - j];
    }
  }
  return G[n]
};


//97https://leetcode.com/problems/interleaving-string/description/

//Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
//Output: true
//Explanation: One way to obtain s3 is:
//Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
//Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = //"aadbbcbcac".



const isInterleave = function (s1, s2, s3) {
  if (s3.length !== s1.length + s2.length) {
    return false;
  }
  const dp = Array.from({ length: s1.length + 1}, () => Array(s2.length + 1).fill(false),);

  for (let i = 0; i <= s1.length; i++) {
    for (let j = 0; j <= s2.length; j++) {
      if (i === 0 && j === 0) {
        dp[i][j] = true;
      } else if (i === 0) {
        dp[i][j] = dp[i][j - 1] && s2[j - 1] === s3[i + j - 1];
      } else if (j === 0) {
        dp[i][j] = dp[i - 1][j] && s1[i - 1] === s3[i + j - 1];
      } else {
        dp[i][j] =
        (dp[i - 1][j] && s1[i - 1] === s3[i + j - 1]) || 
        (dp[i][j - 1] && s2[j - 1] === s3[i + j -1]);
      }
    }
  }
  return dp[s1.length][s2.length];
};


//98 https://leetcode.com/problems/validate-binary-search-tree/

//Input: root = [2,1,3]
//Output: true


const isValidBST = function (root) {
  let prev = -Infinity;
  function inorder (node) {
    if (!node) {
      return true;
    }
    if (!inorder(node.left)) {
      return false;
    }
    if (node.val <= prev) {
      return false;
    }
    prev = node.val;
    return inorder(node.right);
  }
  return inorder(root);
};



//99 https://leetcode.com/problems/recover-binary-search-tree/description/


//Input: root = [1,3,null,null,2]
//Output: [3,1,null,null,2]
//Explanation: 3 cannot be a left child of 1 because 3 > 1. Swapping 1 //and 3 makes the BST valid.


const inorder = function (root, nums) {
  if (!root) return;
  inorder(root.left, nums);
  nums.push(root.val);
  inorder(root.right, nums);
};


const findTwoSwapped = function (nums) {
  let n = nums.length;
  let x = -1,
      y = -1;
      let swappedFirstOccurrence = false;

      for (let i = 0; i < n - 1; ++i) {
        if (nums[i + 1] < nums[i]) {
          y = nums[i + 1];
          if (!swappedFirstOccurrence) {
            x = nums[i];
            swappedFirstOccurrence = true;
          } else {
            break;
          }
        }
      }

      return [x, y];
};

const recover = function (r, count, x, y) {
  if (r) {
    if (r.val === x || r.val === y) {
      r.val = r.val === x ? y : x;
      if (--count === 0) return;
    }
    recover(r.left, count, x, y);
    recover(r.right, count, x, y);
  }
};

const recoverTree = function (root) {
  let nums = [];
  inorder(root, nums);
  let swapped = findTwoSwapped(nums);
  recover(root, 2, swapped[0], swapped[1]);
};



//100 https://leetcode.com/problems/same-tree/description/

//Input: p = [1,2,3], q = [1,2,3]
//Output: true


const isSameTree = function (p, q) {
  if (p == null && q == null) return true;
  if (q == null || p == null) return false;
  if (p.val != q.val) return false;
  return isSameTree(p.right, q.right) && isSameTree(p.left, q.left);;
};


// 101 Symmetric Tree 

//Input: root = [1,2,2,3,4,4,3]
//Output: true

const isSymmetric = function (root) {
  return isMirror(root, root)
};

const isMirror = function (t1, t2) {
  if (!t1 && !t2) return true;
  if (!t1 || !t2) return false;

  return (
    t1.val === t2.val && isMirror(t1.right, t2.left) && isMirror(t1.left, t2.right)
  );
};

//102 https://leetcode.com/problems/binary-tree-level-order-traversal/

//Input: root = [3,9,20,null,null,15,7]
//Output: [[3],[9,20],[15,7]]


const  levelOrder = function (root) {
  let levels = [];
  function helper(node, level) {
    if (levels.length === level) levels.push([]);
    levels[level].push(node.val);
    if (node.left !== null) helper(node.left, level + 1);
    if (node.right !== null) helper(node.right, level + 1);
  }

  if (root !== null) helper(root, 0);
  return levels;
};


//103  https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/



//Input: root = [3,9,20,null,null,15,7]
//Output: [[3],[20,9],[15,7]]


const zigzagLevelOrder = function (root) {
  if (root === null) return [];
  const results = [];
  const node_queue = [root, null];
  const level_list = [];
  let is_order_left = true;
  while (node_queue.length > 0) {
    const curr_node = node_queue.shift();
    if (curr_node !== null) {
      if (is_order_left) level_list.push(curr_node.val);

      else level_list.unshift(curr_node.val);
      if (curr_node.left !== null) node_queue.push(curr_node.left);

      if (curr_node.right !== null) node_queue.push(curr_node.right);
    
    
    } else {
      results.push([...level_list]);
      level_list.length = 0;
      if (node_queue.length > 0) node_queue.push(null);

      is_order_left = !is_order_left;
    }
  }
  return results;
};


//104 https://leetcode.com/problems/maximum-depth-of-binary-tree/


//Input: root = [3,9,20,null,null,15,7]
//Output: 3

const maxDepth = function (root) {
  if (root === null) {
    return 0;
  }

  const left_height = maxDepth(root.left);
  const right_height = maxDepth(root.right);
  return 1 + Math.max(left_height, right_height);
};


//105 https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

//Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
//Output: [3,9,20,null,null,15,7]

const buildTree = function (preorder, inorder) {
  let preorderIndex = 0;
  let inorderIndexMap = new Map();

  for (let i = 0; i < inorder.length; i++) {
    inorderIndexMap.set(inorder[i], i);
  }

  function arrayToTree(left, right) {
    if (left > right) return null;
    let rootValue = preorder[preorderIndex++];
    let root = new TreeNode(rootValue);
    root.left = arrayToTree(left, inorderIndexMap.get(rootValue) - 1);
    root.right = arrayToTree(inorderIndexMap.get(rootValue) + 1, right);
    return root;
  }
  return arrayToTree(0, preorder.length - 1);
};


// 106 https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/


// Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
//Output: [3,9,20,null,null,15,7]


const constructTree = function (inorder, postorder) {
  const idx_map = {};
  const post_idx = postorder.length - 1;
  const helper = function (in_left, in_right) {
    if (in_left > in_right) return null;
    const root_val = postorder[post_idx];
    const root = new TreeNode(root_val);
    const index = idx_map[root_val];
    post_idx--;
    root.right = helper(index + 1, in_right);
    root.left = helper(in_left, index - 1);
    return root;
  };
  for (let i = 0; i < inorder.length; i++) idx_map[inorder[i]] = i;
  return helper(0, inorder.length - 1);
};



// 107 https://leetcode.com/problems/binary-tree-level-order-traversal-ii/

// Input: root = [3,9,20,null,null,15,7]
//Output: [[15,7],[9,20],[3]]

const levelOrderBottom = function (root) {
  let levels = [];
  function helper(node, level) {
    if (!node) return;
    if (!levels[level]) levels[level] = [];
    levels[level].push(node.val);
    if (node.left) helper(node.left, level + 1);
    if (node.right) helper(node.right, level + 1);
  }
  helper(root, 0);
  return levels.reverse();
};




//108 https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/

//  Input: nums = [-10,-3,0,5,9]
//Output: [0,-3,9,-10,null,5]
//Explanation: [0,-10,5,null,-3,null,9] is also accepted:



const sortedArrayToBST = function (num) {
  return helper(nums, 0, nums.length - 1);
};

const helpers = function (nums, left, right) {
  if (left > right) {
    return null;
  }

  const p = Math.floor((left + right) / 2);
  const root = new TreeNode(nums[p], null, null);
  root.left = helpers(nums, left, p - 1);
  root.right = helpers(nums, p + 1, right);
  return root;
};


// 109   https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/


//Input: head = [-10,-3,0,5,9]
//Output: [0,-3,9,-10,null,5]
//Explanation: One possible answer is [0,-3,9,-10,null,5], which //represents the shown height balanced BST.


function ListNode (val, next) {
  this.val = (val === undefined ? 0 : val);
  this.next = (next === undefined ? null : next);
}

function TreeNode(val, left, right) {
  this.val = (val === undefined ? 0 : val);
  this.left = (left === undefined ? null : left);
  this.right = (right === undefined ? null : right);
}

const sortedListToBST = function(head) {
  if (!head) return null;
  const mid = findMiddleElement(head);
  const node = new TreeNode(mid.val);
  if (head === mid) return node;
  node.left = sortedListToBST(head);
  node.right = sortedListToBST(mid.next);
  return node;
};


const findMiddleElement = function(head) {
  const prevPtr = null;
  const slowPtr = head;
  const fastPtr = head;
  while (fastPtr && fastPtr.next) {
    prevPtr = slowPtr;
    slowPtr = slowPtr.next;
    fastPtr = fastPtr.next.next;
  }

  if (prevPtr != null) prevPtr.next = null;
  return slowPtr;
};

// 110  https://leetcode.com/problems/balanced-binary-tree/


//  Input: root = [3,9,20,null,null,15,7]
//Output: true


const height = function (root) {
  if (root == null) {
    return -1;
  }
  return 1 + Math.max(height(root.left), height(root.right));
};

const isBalanced = function (root) {
  if (root == null) {
    return true;
  }
  return (
    Math.abs(height(root.left) - height(root.right)) < 2 &&

    isBalanced(root.left) &&
    isBalanced(root.right)
  );
};

// 111 https://leetcode.com/problems/minimum-depth-of-binary-tree/description/

//  Input: root = [3,9,20,null,null,15,7]
//Output: 2

const minDepth = function (root) {
  function dfs(root) {
    if (root === null) {
      return 0;
    }
    if (root.left === null) {
      return 1 + dfs(root.right);
    } else if (root.right === null) {
      return 1 + dfs(root.left);
    }
    return 1 + Math.min(dfs(root.left), dfs(root.right));
  }
  return dfs(root);

};


//112 https://leetcode.com/problems/path-sum/


//Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
//Output: true
//Explanation: The root-to-leaf path with the target sum is shown.



const hasPathSum = function (root, sum) {
  if (!root) return false;
  let nodeStack = [];
  let sumStack = [];
  nodeStack.push(root);
  sumStack.push(sum - root.val);
  while (nodeStack.length > 0) {
    let currentNode = nodeStack.pop();
    let currSum = sumStack.pop();
    if (!currentNode.left && !currentNode.right && currSum === 0)
      return true;
    if (currentNode.right) {
      nodeStack.push(currentNode.right);
      sumStack.push(currSum - currentNode.right.val);
    }
    if (currentNode.left) {
      nodeStack.push(currentNode.left);
      sumStack.push(currSum - currentNode.left.val);
    }
  }
  return false;
};


// 113 https://leetcode.com/problems/path-sum-ii/description/


//Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
//Output: [[5,4,11,2],[5,8,4,5]]
//Explanation: There are two paths whose sum equals targetSum:
//5 + 4 + 11 + 2 = 22
//5 + 8 + 4 + 5 = 22


const pathSum = function (root, sum) {
  let pathList = [];
  let pathNodes = [];
  let recurseTree = function (node, remainingSum, pathNodes, pathList) {
    if (!node) {
      return;
    }
    pathNodes.push(node.val);
    if (
      remainingSum === node.val && 
      node.left === null &&
      node.right === null
    ) {
      pathList.push(Array.from(pathNodes));
    } else {
      recurseTree(
        node.left,
        remainingSum - node.val,
        pathNodes,
        pathList,
      );
      recurseTree(
        node.right,
        remainingSum - node.val,
        pathNodes,
        pathList
      );
    }
    pathNodes.pop();
  };
  recurseTree(root, sum, pathNodes, pathList);
  return pathList;
};

// 114 https://leetcode.com/problems/flatten-binary-tree-to-linked-list/

//Input: root = [1,2,5,3,4,null,6]
//Output: [1,null,2,null,3,null,4,null,5,null,6]

const flattenTree = function (node) {
  if (node == null) {
    return null;
  }
  if (node.left == null && node.right == null) {
    return node;
  }
  let leftTail = flattenTree(node.left);
  let rightTail = flattenTree(node.right);
  if (leftTail != null) {
    leftTail.right = node.right;
    node.right = node.left;
    node.left = null;
  }
  return rightTail == null ? leftTail : rightTail;
};
const flatten = function (root) {
  flattenTree(root);
 
};




//115 https://leetcode.com/problems/distinct-subsequences/

//Input: s = "rabbbit", t = "rabbit"
//Output: 3
//Explanation:
//As shown below, there are 3 ways you can generate "rabbit" from s.
//rabbbit
//rabbbit
//rabbbit


const numDistinct = function (s, t) {
  let memo = new Map();
  function dp(i, j) {
    if (i === s.length || j === t.length || s.length - i < t.length - j)
      return j === t.length ? 1 : 0;

    let key = [i, j].toString();
    if (memo.has(key)) return memo.get(key);
    let ans = dp(i + 1, j);
    if (s[i] === t[j]) ans += dp(i + 1, j + 1);
    memo.set(key, ans);
    return ans;
  }
  return dp(0, 0);
};


// 116 https://leetcode.com/problems/populating-next-right-pointers-in-each-node/description/


//struct Node {
  //int val;
  //Node *left;
  //Node *right;
  //Node *next;
//}

//Input: root = [1,2,3,4,5,6,7]
//Output: [1,#,2,3,#,4,5,6,7,#]
//Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as //connected by the next pointers, with '#' signifying the end of each //level.


//while (!Q.empty())
//{
  //size = Q.size()
  //for i in range 0..size
 // {
      //node = Q.pop()
      //Q.push(node.left)
      //Q.push(node.right)
  //}
//}


const connect = function (root) {
  if (root == null) {
    return root;
  }

  let Q = [];
  Q.push(root);
  while (Q.length > 0) {
    let size = Q.length;
    for (let i = 0; i < size; i++) {
      let node = Q.shift();
      if (i < size - 1) {
        node.next = Q[0];
      }
      if (node.left != null) {
        Q.push(node.left);
      }
      if (node.right != null) {
        Q.push(node.right);
      }
    }
  }
  return root;
};


// 117 https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/editorial/


// Input: root = [1,2,3,4,5,null,7]
//Output: [1,#,2,3,#,4,5,7,#]
//Explanation: Given the above binary tree (Figure A), your function //should populate each next pointer to point to its next right node, just //like in Figure B. The serialized output is in level order as connected //by the next pointers, with '#' signifying the end of each level.


const Connect = function (root) {
  if (!root) return root;
  const Q = [];
  Q.push(root);
  while (Q.length > 0) {
    const size = Q.length;
    for (let i = 0; i < size; i++) {
      const node = Q.shift();
      if (i < size - 1) {
        node.next = Q[0];
      }
      if (node.left) Q.push(node.left);
      if (node.right) Q.push(node.right);
    }
  }
  return root;
};


// 118 https://leetcode.com/problems/pascals-triangle/editorial/


//Input: numRows = 5
//Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]


const generate = function (numRows) {
  const triangle = [];
  triangle.push([]);
  triangle[0].push(1);
  for (let rowNum = 1; rowNum < numRows; rowNum++) {
    const row = [];
    const prevRow = triangle[rowNum - 1];
    row.push(1);
    for (let j = 1; j < rowNum; j++) {
      row.push(prevRow[j - 1] + prevRow[j]);
    }
    row.push(1);
    triangle.push(row);
  }
  return triangle;
}


// 119  https://leetcode.com/problems/pascals-triangle-ii/editorial/


//Input: rowIndex = 3
//Output: [1,3,3,1]


const getNum = function (row, col) {
  if (row === 0 || col === 0 || row === col) {
    return 1;
  }
  return getNum(row - 1, col - 1) + getNum(row - 1, col);
};

const getRow = function (rowIndex) {
  let ans = [];
  for (let i = 0; i <= rowIndex; i++) {
    ans.push(getNum(rowIndex, i));
  }
  return ans;
};


//120 https://leetcode.com/problems/triangle/description/


//Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
//Output: 11
//Explanation: The triangle looks like:
   //2
  //3 4
 //6 5 7
//4 1 8 3
//The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 //(underlined above).


const minimumTotal = function (triangle) {
  for (let row = 1; row < triangle.length; row++) {
    for (let col = 0; col <= row; col++) {
      let smallestAbove = Number.MAX_VALUE;
      if (col > 0) {
        smallestAbove = triangle[row - 1][col - 1];
      }
      if (col < row) {
        smallestAbove = Math.min(smallestAbove, triangle[row - 1][col]);
      }
      triangle[row][col] += smallestAbove;
    }
  }
  return Math.min(...triangle[triangle.length - 1]);
};


//121  https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/


//Input: prices = [7,1,5,3,6,4]
//Output: 5
//Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), //profit = 6-1 = 5.
//Note that buying on day 2 and selling on day 1 is not allowed because //you must buy before you sell.


const maxProfit = function (prices) {
  let minprice = Number.MAX_VALUE;
  let maxprofit = 0;
  for (let i = 0; i < prices.length; i++) {
    if (prices[i] < minprice) minprice = prices[i];
    else if (prices[i] - minprice > maxprofit)
      maxprofit = prices[i] - minprice;
  }
  return maxprofit;
};


//122 https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/


//Input: prices = [7,1,5,3,6,4]
//Output: 7
//Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), //profit = 5-1 = 4.
//Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = //6-3 = 3.
//Total profit is 4 + 3 = 7.


const maxBestProfit = function (prices) {
  let i = 0;
  let valley = prices[0];
  let peak = prices[0];
  let maxprofit = 0;
  while (i < prices.length - 1) {
    while (i < prices.length - 1 && prices[i] >= prices[i + 1]) i++;
    valley = prices[i];
    while (i < prices.length - 1 && prices[i] <= prices[i + 1]) i++;

    peak = prices[i];
    maxprofit += peak - valley;
  }
  return maxprofit;
};


//123  https://leetcode.com/problems/binary-tree-maximum-path-sum/

//Input: prices = [3,3,5,0,0,3,1,4]
//Output: 6
//Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), //profit = 3-0 = 3.
//Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = //4-1


const maxPathProfit = function (prices) {
  if (prices.length <= 1) return 0;
  let left_min = prices[0];
  let right_max = prices[prices.length - 1];
  let length = prices.length;
  let left_profits = new Array(length).fill(0);
  let right_profits = new Array(length + 1).fill(0);
  for (let l = 1; l < length; ++l) {
    left_profits[l] = Math.max(left_profits[l - 1], prices[l] - left_min);
    left_min = Math.min(left_min, prices[l]);
    let r = length - 1 - l;
    right_profits[r] = Math.max(right_profits[r + 1], right_max - prices[r],);
    right_max = Math.max(right_max, prices[r]);
  }
  let max_profit = 0;
  for (let i = 0; i < length; ++i) {
    max_profit = Math.max(max_profit,
      left_profits[i] + right_profits[i + 1],
    );
  }
  return max_profit;
};


// 125 https://leetcode.com/problems/valid-palindrome/

//Input: s = "A man, a plan, a canal: Panama"
//Output: true
//Explanation: "amanaplanacanalpanama" is a palindrome.


const isPalindrome = function (s) {
  let i = 0;
  let j = s.length - 1;
  while (i < j) {
    while (i < j && !isLetterOrDigit(s[i])) {
      i++;
    }
    while (i < j && !isLetterOrDigit(s[j])) {
      j--;
    }
    if ((s[i] + "").toLowerCase() !== (s[j] + "").toLowerCase())
      return false;
    i++;
    j--;
  }
  return true;
};

function isLetterOrDigit(character) {
  const charCode = character.charCodeAt();
  return (
    (charCode >= "a".charCodeAt() && charCode <= "z".charCodeAt()) ||
    (charCode >= "A".charCodeAt() && charCode <= "Z".charCodeAt()) ||
    (charCode >= "0".charCodeAt() && charCode <= "9".charCodeAt())
  );
}

// 126 https://leetcode.com/problems/word-ladder-ii/

//Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot",//"dog","lot","log","cog"]
//Output: [["hit","hot","dot","dog","cog"],["hit","hot","lot","log",//"cog"]]
//Explanation: There are 2 shortest transformation sequences:
//"hit" -> "hot" -> "dot" -> "dog" -> "cog"
//"hit" -> "hot" -> "lot" -> "log" -> "cog"


const findLadders = function(beginWord, endWord, wordList) {
  let adjList = {};
  let currPath = [endWord];
  let shortestPaths = [];
  let wordSet = new Set(wordList);

  function findNeighbors(word) {
    let neighbors = [];
    let charList = Array.from(word);
    for (let i = 0; i < charList.length; i++) {
      let oldChar = charList[i];
      for (let c = 'a'.charCodeAt(0); c <= 'z'.charCodeAt(0); c++) {
        if (c !== oldChar.charCodeAt(0)) {
          charList[i] = String.fromCharCode(c);
          let newWord = charList.join("");
          if (wordSet.has(newWord)) neighbors.push(newWord);
        }
      }
      charList[i] = oldChar;
    }
    return neighbors;
  }

  function backtrack(source) {
    if (source === endWord) {
      shortestPaths.push([...currPath].reverse());
      return;
    }
    adjList[source]?.forEach(neighbor => {
      currPath.push(neighbor);
      backtrack(neighbor);
      currPath.pop();
    });
  }
  function bfs() {
    let queue = [beginWord];
    wordSet.delete(beginWord);
    while (queue.length) {
      let current = queue.shift();
      let neighbors = findNeighbors(current);
      neighbors.forEach(neighbor => {
        if (!adjList[neighbor]) adjList[neighbor] = [];

        adjList[neighbor].push(current);
        if (!wordSet.has(neighbor)) {
          queue.push(neighbor);
          wordSet.delete(neighbor);
        }
      });
    }
  }
  bfs();
  backtrack(beginWord);
  return shortestPaths;
};

//127 https://leetcode.com/problems/word-ladder/


//Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot",//"dog","lot","log","cog"]
//Output: 5
//Explanation: One shortest transformation sequence is "hit" -> "hot" -> //"dot" -> "dog" -> cog", which is 5 words long.


const ladderLength = function (beginWord, endWord, wordList) {
  let L = beginWord.length;
  let allComboDict = {};
  for (let word of wordList) {
    for (let i = 0; i < L; i++) {
      let newWord = word.substring(0, i) + "*" + word.substring(i + 1, L);
      if (!allComboDict[newWord]) allComboDict[newWord] = [];

      allComboDict[newWord].push(word);
    }
  }
  let Q = [[beginWord, 1]];
  let visited = { [beginWord]: true };
  while (Q.length > 0) {
    let node = Q.shift();
    let word = node[0];
    let level = node[1];
    for (let i = 0; i < L; i++) {
      let newWord = word.substring(0, i) + "*" + word.substring(i + 1, L);
      for (let adjacentWord of allComboDict[newWord] || []) {
        if (adjacentWord === endWord) {
          return level + 1;
        }
        if (!visited[adjacentWord]) {
          visited[adjacentWord] = true;
          Q.push([adjacentWord, level + 1]);
        }
      }
    }
  }
  return 0;
};

//128 https://leetcode.com/problems/longest-consecutive-sequence/description/

// Input: nums = [100,4,200,1,3,2]
//Output: 4
//Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. //Therefore its length is 4.


const longestConsecutive = function (nums) {
  let longestStreak = 0;
  for (let num of nums) {
    let currentNum = num;
    let currentStreak = 1;
    while (nums.includes(currentNum + 1)) {
      currentNum += 1;
      currentStreak += 1;
    }
    longestStreak = Math.max(longestStreak, currentStreak);
  }
  return longestStreak;
};


//129 https://leetcode.com/problems/sum-root-to-leaf-numbers/description/


//Input: root = [1,2,3]
//Output: 25
//Explanation:
//The root-to-leaf path 1->2 represents the number 12.
//The root-to-leaf path 1->3 represents the number 13.
//Therefore, sum = 12 + 13 = 25.


function sumNumbers(root, partialSum = 0) {
  if (!root) {
    return 0;
  }
  partialSum = partialSum * 10 + root.val;
  if (!root.left && !root.right) {
    return partialSum;
  }
  return (
    sumNumbers(root.left, partialSum) + sumNumbers(root.right, partialSum)
  );
}


//130 https://leetcode.com/problems/surrounded-regions/


//Input: board = [["X"]]

//Output: [["X"]]


const solve = function (board) {
  if (board ==null || board.length == 0) {
    return;
  }

  let ROWS = board.length;
  let COLS = board[0].length;
  let borders = [];
  for (let r = 0; r < ROWS; ++r) {
    borders.push([r, 0]);
    borders.push([r, COLS - 1]);
  }
  for (let c = 0; c < COLS; ++c) {
    borders.push([0, c]);
    borders.push([ROWS - 1, c]);
  }

  borders.forEach((pair) => {
    dfs(board, pair[0], pair[1]);
  });

  for (let r = 0; r < ROWS; ++r) {
    for (let c = 0; c < COLS; ++c) {
      if (board[r][c] == "O") board[r][c] = "X";
      if (board[r][c] == "E") board[r][c] = "O";
    }
  }

  function dfs(board, row, col) {
    if (board[row][col] != "O") return;
    board[row][col] = "E";
    if (col < COLS - 1) dfs(board, row, col + 1);
    if (row < ROWS - 1) dfs(board, row + 1, col);
    if (col > 0) dfs(board, row, col - 1);
    if (row > 0) dfs(board, row - 1, col);
  }
};


//131  https://leetcode.com/problems/palindrome-partitioning/description/

//Input: s = "aab"
//Output: [["a","a","b"],["aa","b"]]


const Partition = function (s) {
  const res = [];
  dfs(s, [], res);
  return res;
  function dfs(s, path, res) {
    if (!s.length) {
      res.push(path);
      return;
    }
    for (let i = 0; i < s.length; i++) {
      const cur = s. substr(0, i + 1);
      if (isPalindrome(cur)) {
        dfs(s.substr(i + 1), path.concat(cur), res);
      }
    }
  }
  function isPalindrome(s) {
    let lo = 0,
    hi = s.length - 1;
    while (lo < hi) {
      if (s[lo++] != s[hi--]) return false;
    }
    return true;
  }
};


//132 https://leetcode.com/problems/palindrome-partitioning-ii/

// Input: s = "aab"
//Output: 1
//Explanation: The palindrome partitioning ["aa","b"] could be produced //using 1 cut.


const minCut = function (s) {
  return findMinimumCut(s, 0, s.length - 1, s.length - 1);
};

const findMinimumCut = function (s, start, end, minimumCut) {
  if (start === end || Palindrome(s, start, end)) {
    return 0;
  }
  for (let currentEndIndex = start; currentEndIndex <= end; currentEndIndex++) {
    if (Palindrome(s, start, currentEndIndex)) {
      minimumCut = Math.min(
        minimumCut,
        1 + findMinimumCut(s, currentEndIndex + 1, end, minimumCut),
      );
    }
  }
  return minimumCut;
};

const Palindrome = function (s, start, end) {
  while (start < end) {
    if (s[start++] !== s[end--]) {
      return false;
    }
  }
  return true;
};


// 133 https://leetcode.com/problems/clone-graph/description/


//Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
//Output: [[2,4],[1,3],[2,4],[1,3]]
//Explanation: There are 4 nodes in the graph.
//1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val //= 4).
//2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val //= 3).
//3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val //= 4).
//4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).


const cloneGraph = function (node) {
  if (node === null) return node;
  const visited = new Map();
  const queue = [node];
  visited.set(node, {val: node.val, neighbors: []});

  while (queue.length > 0) {
    const n = queue.shift();
    n.neighbors.forEach((neighbor) => {
      if (!visited.has(neighbor)) {
        visited.set(neighbor, {val: neighbor.val, neighbors: []});
        queue.push(neighbor);
      }
      visited.get(n).neighbors.push(visited.get(neighbor));
    });
  }
  return visited.get(node);
};


//134 https://leetcode.com/problems/gas-station/


// Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
//Output: 3
//Explanation:
//Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank //=// 0 + 4 = 4
//Travel to station 4. Your tank = 4 - 1 + 5 = 8
//Travel to station 0. Your tank = 8 - 2 + 1 = 7
//Travel to station 1. Your tank = 7 - 3 + 2 = 6
//Travel to station 2. Your tank = 6 - 4 + 3 = 5
//Travel to station 3. The cost is 5. Your gas is just enough to travel //back to station 3.
//Therefore, return 3 as the starting index.


const canCompleteCircuit = function (gas, cost) {
  let currGain = 0,
  totalGain = 0,
  answer = 0;
  for (let i = 0; i < gas.length; ++i) {
    totalGain += gas[i] - cost[i];
    currGain += gas[i] - cost[i];
    if (currGain < 0) {
      answer = i + 1;
      currGain = 0;
    }
  }
  return totalGain >= 0 ? answer : -1;
};


//135 https://leetcode.com/problems/candy/

//Input: ratings = [1,0,2]
//Output: 5
//Explanation: You can allocate to the first, second and third child with //2, 1, 2 candies respectively.


const candy = function (ratings) {
  let sum = 0;
  let len = ratings.length;
  let left2right = new Array(len).fill(1);
  let right2left = new Array(len).fill(1);
  for (let i = 1; i < len; i++) {
    if (ratings[i] > ratings[i - 1]) {
      left2right[i] = left2right[i + 1] + 1;
    }
  }
  for (let i = 0; i < len; i++) {
    sum += Math.max(left2right[i], right2left[i]);
  }
  return sum;
};


//136 https://leetcode.com/problems/single-number-ii/


//Input: nums = [2,2,3,2]
//Output: 3


const singleNumber = function (nums) {
  nums.sort();
  for (let i = 0; i < nums.length - 1; i += 3) {
    if (nums[i] == nums[i + 1]) {
      continue;
    } else {
      return nums[i];
    }
  }
  return nums[nums.length - 1];
};


//138 https://leetcode.com/problems/copy-list-with-random-pointer/


//Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
//Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]


function Node(val, next, random) {
  this.val = val;
  this.next = next;
  this.random = random;
}

const copyRandomList = function (head) {
  let visitedHash = new Map();
  let cloneNode = function (node) {
    if (node === null) {
      return null;
    }
    if (visitedHash.has(node)) {
      return visitedHash.get(node);
    }
    let newNode = new Node(node.val, null, null);
    visitedHash.set(node, newNode);
    newNode.next = cloneNode(node.next);
    newNode.random = cloneNode(node.random);
    return newNode;
  };
  return cloneNode(head);
};


//139 https://leetcode.com/problems/word-break/description/


// Input: s = "leetcode", wordDict = ["leet","code"]
//Output: true
//Explanation: Return true because "leetcode" can be segmented as "leet //code".


const wordBreak = function (s, wordDict) {
  let words = new Set(wordDict);
  let queue = [0];
  let seen = new Set();
  while (queue.length != 0) {
    let start = queue.shift();
    if (start == s.length) return true;
    for (let end = start + 1; end <= s.length; end++) {
      if (seen.has(end)) continue;
      if (words.has(s.substring(start, end))) {
        queue.push(end);
        seen.add(end);
      }
    }
  }
  return false;
};


//140 https://leetcode.com/problems/word-break-ii/


//Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
//Output: ["cats and dog","cat sand dog"]



class Solution {
  wordBreak(s, wordDict) {
    const wordSet = new Set(wordDict);
    const results = [];
    this._backtrack(s, wordSet, [], results, 0);
    return results;
  }

  _backtrack(s, wordSet, currentSentence, results, startIndex) {
    if (startIndex === s.length) {
      results.push(currentSentence.join(" "));
      return;
    }
    for (let endIndex = startIndex + 1; endIndex <= s.length; endIndex++) {
      const word = s.substring(startIndex, endIndex);
      if (wordSet.has(word)) {
        currentSentence.push(word);
        this._backtrack(s, wordSet, currentSentence, results, endIndex);
        currentSentence.pop();
      }
    }
  }
}



//141 https://leetcode.com/problems/linked-list-cycle/


//Input: head = [3,2,0,-4], pos = 1
//Output: true
//Explanation: There is a cycle in the linked list, where the tail 


const hasCycle = function (head) {
  let nodesSeen = new Set();
  let current = head;
  while (current != null) {
    if (nodesSeen.has(current)) {
      return true;
    }
    nodesSeen.add(current);
    current = current.next;
  }
  return false;
};



//142 https://leetcode.com/problems/linked-list-cycle-ii/


//Input: head = [3,2,0,-4], pos = 1
//Output: tail connects to node index 1
//Explanation: There is a cycle in the linked list, where tail connects to the second node.


function ListNode(val, next) {
  this.val = (val === undefined ? 0 : val);
  this.next = (next === undefined ? null : next);
}

const detectCycle = function(head) {
  let nodesSeen = new Set();
  let node = head;
  while (node !== null) {
    if (nodesSeen.has(node)) {
      return node;
    } else {
      nodesSeen.add(node);
      node = node.next;
    }
  }
  return null;
};


// 143 https://leetcode.com/problems/reorder-list/

//Input: head = [1,2,3,4]
//Output: [1,4,2,3]


const reorderList = function (head) {
  if (head === null) return;
  let slow = head;
  let fast = head;
  while (fast !== null && fast.next !== null) {
    slow = slow.next;
    fast = fast.next.next;
  }
  let prev = null;
  let curr = slow;
  while (curr !== null) {
    let tmp = curr.next;
    curr.next = prev;
    prev = curr;
    curr = tmp;
  }
  let first = head;
  let second = prev;
  while (second.next !== null) {
    let tmp = first.next;
    first.next = second;
    first = tmp;
    tmp = second.next;
    second.next = first;
    second = tmp;
  }
};


//144 https://leetcode.com/problems/binary-tree-preorder-traversal/


// Input: root = [1,null,2,3]
//Output: [1,2,3]


const preorderTraversal = function (root) {
  if (!root) {
    return [];
  }

  const stack = [root];
  const output = [];

  while (stack.length != 0) {
    root = stack.pop();
    if (root) {
      output.push(root.val);
      if (root.right) {
        stack.push(root.right);
      }
      if (root.left) {
        stack.push(root.left);
      }
    }
  }
  return output;
};



//145 https://leetcode.com/problems/binary-tree-postorder-traversal/


//Input: root = [1,null,2,3]
//Output: [3,2,1]


const postorderTraversal = function (root) {
  let output = [];
  let stack = [];
  if (root === null) return output;
  stack.push(root);
  while (stack.length > 0) {
    root = stack.pop();
    output.push(root.val);
    if (root.left !== null) stack.push(root.left);
    if (root.right !== null) stack.push(root.right);
  }
  output.reverse();
  return output;
};



//146 https://leetcode.com/problems/lru-cache/

// Input
//["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
//[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
//Output
//[null, null, null, 1, null, -1, null, -1, 3, 4]


class Node {
  constructer(key, value) {
    this.key = key;
    this.value = value;
    this.next = null;
    this.prev = null;
  }
}

class LRUCache {
  constructer(capacity) {
    this.capacity = capacity;
    this.dic = new Map();
    this.head = new Node(-1, -1);
    this.tail = new Node(-1, -1);
    this.head.next = this.tail;
    this.tail.prev = this.head;
  }

  get(key) {
    if (!this.dic.has(key)) {
      return -1;
    }

    let node = this.dic.get(key);
    this.remove(node);
    this.add(node);
    return node.value;
  }

  put(key, value) {
    if (this.dic.has(key)) {
      this.remove(this.dic.get(key));
    }
    let node = new Node(key, value);
    this.add(node);
    this.dic.set(key, node);
    if (this.dic.size > this.capacity) {
      let nodeToDelete = this.head.next;
      this.remove(nodeToDelete);
      this.dic.delete(nodeToDelete.key);
    }
  }
  add(node) {
    let pre = this.tail.prev;
    pre.next = node;
    node.prev = pre;
    node.next = this.tail;
    this.tail.prev = node;
  }
  remove(node) {
    let pre = node.prev;
    let nxt = node.next;
    pre.next = nxt;
    nxt.prev = pre;
  }
}



//147 https://leetcode.com/problems/insertion-sort-list/


//Input: head = [4,2,1,3]
//Output: [1,2,3,4]


var insertionSortList = function (head) {
  let dummy = new ListNode();
  let curr = head;
  while (curr !== null) {
      let prev = dummy;
      while (prev.next !== null && prev.next.val <= curr.val) {
          prev = prev.next;
      }
      let next = curr.next;
      curr.next = prev.next;
      prev.next = curr;
      curr = next;
  }
  return dummy.next;
};



//148 https://leetcode.com/problems/sort-list/


//Input: head = [4,2,1,3]
//Output: [1,2,3,4]


const sortList = function (head) {
  if (!head || !head.next) return head;

  let mid = getMid(head);
  let left = sortList(head);
  let right = sortList(mid);
  return merge(left, right);


  function merge(list1, list2) {
    let dummyHead = new ListNode();
    let tail = dummyHead;
    while (list1 && list2) {
      if (list1.val < list2.val) {
        tail.next = list1;
        list1 = list1.next;
      } else {
        tail.next = list2;
        list2 = list2.next;
      }
      tail = tail.next;
    }
    tail.next = list1 ? list1 : list2;
    return dummyHead.next;
  }

  function getMid(head) {
    let midPrev = null;
    while (head && head.next) {
      midPrev = midPrev === null ? head : midPrev.next;
      head = head.next.next;
    }
    let mid = midPrev.next;
    midPrev.next = null;
    return mid;
  }
};



// 149 https://leetcode.com/problems/max-points-on-a-line/


//Input: points = [[1,1],[2,2],[3,3]]
//Output: 3


const maxPoints = function (points) {
  let n = points.length;
  if (n == 1) {
    return 1;
  }

  let result = 2;
  for (let i = 0; i < n; i++) {
    let cnt = {};
    for (let j = 0; j < n; j++) {
      if (j != i) {
        let key = Math.atan2(
          points[j][1] - points[i][1],
          points[j][0] - points[i][0],
        );
        cnt[key] = cnt[key] ? cnt[key] + 1 : 1;
      }
    }
    result = Math.max(result, Math.max(...Object.values(cnt)) + 1);
  }
  return result;
};



//150 https://leetcode.com/problems/evaluate-reverse-polish-notation/


//Input: tokens = ["2","1","+","3","*"]
//Output: 9
//Explanation: ((2 + 1) * 3) = 9


const OPERATORS = {
  "+": (a, b) => a + b,
  "-": (a, b) => a - b,
  "/": (a, b) => Math.trunc(a / b),
  "*": (a, b) => a * b,
};

const evalRPN = function (tokens) {
  let currentPosition = 0;

  while (tokens.length > 1) {
     
    while (!(tokens[currentPosition] in OPERATORS)) {
      currentPosition += 1;
    }

    const operator = tokens[currentPosition];
    const operation = OPERATORS[operator];
    const number1 = Number(tokens[currentPosition - 2]);
    const number2 = Number(tokens[currentPosition - 1]);

    tokens[currentPosition] = operation(number1, number2);

    tokens.splice(currentPosition - 2, 2);
    currentPosition -= 1;
  }
  return tokens[0];
};



//151 https://leetcode.com/problems/reverse-words-in-a-string/


//Input: s = "the sky is blue"
//Output: "blue is sky the"


const reverseWords = function (s) {
  s = s.trim();
  let words = s.split(/\s+/).reverse();
  return words.join(" ");
};


//152 https://leetcode.com/problems/maximum-product-subarray/

//Input: nums = [2,3,-2,4]
//Output: 6
//Explanation: [2,3] has the largest product 6.


const maxProduct = function (nums) {
  if (nums.length === 0) return 0;

  let result = nums[0];

  for (let i = 0; i < nums.length; i++) {
    let accu = 1;
    for (let j = i; j < nums.length; j++) {
      accu *= nums[j];
      result = Math.max(result, accu);
    }
  }
  return result;
};


// 153 https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/

//Input: nums = [3,4,5,1,2]
//Output: 1
//Explanation: The original array was [1,2,3,4,5] rotated 3 times.


const findMin = function (nums) {
  if (nums.length == 1) {
    return nums[0];
  }
  if (nums[right] > nums[0]) {
    return nums[0];
  }

  while (right >= left) {
    let mid = left + Math.floor((right - left) / 2);
    if (nums[mid] > nums[mid + 1]) {
      return nums[mid + 1];
    }
    if (nums[mid - 1] > nums[mid]) {
      return nums[mid];
    }
    if (nums[mid] > nums[0]) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return Number.MAX_VALUE;
};



//154 https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/

//Input: nums = [1,3,5]
//Output: 1


const findMinim = function (nums) {
   let low = 0,
   high = nums.length - 1;


  while (low < high) {
    let pivot = low + Math.floor((high - low) / 2);
    if (nums[pivot] < nums[high]) high = pivot;
    else if (nums[pivot] > nums[high]) low = pivot + 1;
    else high -= 1;
  }
  return nums[low];
};


//155 https://leetcode.com/problems/min-stack/description/


//Input
//["MinStack","push","push","push","getMin","pop","top","getMin"]
//[[],[-2],[0],[-3],[],[],[],[]]

//Output
//[null,null,null,null,-3,null,0,-2]

//Explanation
//MinStack minStack = new MinStack();
//minStack.push(-2);
//minStack.push(0);
//minStack.push(-3);
//minStack.getMin(); // return -3
//minStack.pop();
//minStack.top();    // return 0
//minStack.getMin(); // return -2




function last(arr) {
  return arr[arr.length - 1];
}

class MinStack {
  _stack = [];

  push(x) {
    if (this._stack.length === 0) {
      this._stack.push([x, x]);
      return;
    }

    const currentMin = last(this._stack)[1];
    this._stack.push([x, Math.min(currentMin, x)]);
  }

  pop() {
    this._stack.pop();
  }

  top() {
    return last(this._stack)[0];
  }

  getMin() {
    return last(this._stack)[1];
  }
}


//156 https://leetcode.com/problems/read-n-characters-given-read4/

// Input: file = "abc", n = 4
//Output: 3
//Explanation: After calling your read method, buf should contain "abc". We read a total of 3 characters from the file, so return 3.
//Note that "abc" is the file's content, not buf. buf is the destination buffer that you will have to write the results to.


class Reader4 {
  read4(buf4) {
    return 0;
  }
}

class Solution extends Reader4 {
  read(buf, n) {
    let copiedChars = 0, readChars = 4;
    let buf4 = new Array(4).fill('\0');

    while (copiedChars < n && readChars === 4) {
      readChars = this.read4(buf4);

      for (let i = 0; i < readChars; ++i) {
        if (copiedChars === n) return copiedChars;
        buf[copiedChars] = buf4[i];
        ++copiedChars;
      }
    }
    return copiedChars;
  }
}



//160  https://leetcode.com/problems/intersection-of-two-linked-lists/description/


//Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
//Output: Intersected at '8'
//Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect).
//From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,6,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.
//- Note that the intersected node's value is not 1 because the nodes with value 1 in A and B (2nd node in A and 3rd node in B) are different node references. In other words, they point to two different locations in memory, while the nodes with value 8 in A and B (3rd node in A and 4th node in B) point to the same location in memory.



const getIntersectionNode = function (headA, headB) {
  let nodesInB = new Set();

  while (headB !== null) {
    nodesInB.add(headB);
    headB = headB.next;
  }

  while (headA !== null) {
    if (nodesInB.has(headA)) {
      return headA;
    }
    headA = headA.next;
  }
  return null;
};


//161 https://leetcode.com/problems/one-edit-distance/description/


//Input: s = "ab", t = "acb"
//Output: true
//Explanation: We can insert 'c' into s to get t.


const isOneEditDistance = function (s, t) {
  let ns = s.length;
  let nt = t.length;
  if (ns > nt) return isOneEditDistance(t, s);
  if (nt - ns > 1) return false;

  for (let i = 0; i < ns; i++)
    if (s[i] != t[i])
      if (ns == nt) 
        return s.slice(i + 1) === t.slice(i + 1);
      else return s.slice(i) === t.slice(i + 1);
      return ns + 1 === nt;
};


//162 https://leetcode.com/problems/find-peak-element/


//Input: nums = [1,2,3,1]
//Output: 2
//Explanation: 3 is a peak element and your function should return the index number 2.


const findPeakElement = function (nums) {
  return search(nums, 0, nums.length - 1);
};

const searches = function (nums, l, r) {
  if (l == r) return l;
  let mid = Math.floor((l + r) / 2);
  if (nums[mid] > nums[mid + 1]) return searches(nums, l, mid);

  return searches(nums, mid + 1, r);
};


//163 https://leetcode.com/problems/missing-ranges/

//Input: nums = [-1], lower = -1, upper = -1
//Output: []
//Explanation: There are no missing ranges since there are no missing numbers.



const findMissingRanges = function (nums, lower, upper) {
  let n = nums.length;
  let missingRanges = [];
  if (n === 0) {
    missingRanges.push([lower, upper]);
    return missingRanges;
  }

  if (lower < nums[0]) {
    missingRanges.push([lower, nums[0] - 1]);
  }

  for (let i = 0; i < n - 1; i++) {
    if (nums[i + 1] - nums[i] <= 1) {
      continue;
    }
    missingRanges.push([nums[i] + 1, nums[i + 1] - 1]);
  }

  if (upper > nums[n - 1]) {
    missingRanges.push([nums[n - 1] + 1, upper]);
  }

  return missingRanges;
};



//164 https://leetcode.com/problems/maximum-gap/


//  Input: nums = [3,6,9,1]
//Output: 3
//Explanation: The sorted form of the array is [1,3,6,9], either (3,6) or (6,9) has the maximum difference 3.


const maximumGap = function (nums) {
  if (nums == null || nums.length < 2)
    return 0;

  nums.sort((a, b) => a - b)

  let maxGap = 0;

  for (let i = 0; i < nums.length - 1; i++)
    maxGap = Math.max(nums[i + 1] - nums[i], maxGap);

  return maxGap;
};


//165 https://leetcode.com/problems/compare-version-numbers/


//Input: version1 = "1.2", version2 = "1.10"
//Output: -1
//Explanation:
//version1's second revision is "2" and version2's second revision is "10": 2 < 10, so version1 < version2.


const compareVersion = function (version1, version2) {
  let nums1 = version1.split(".");
  let nums2 = version2.split(".");
  let n1 = nums1.length,
     n2 = nums2.length;

     for (let i = 0; i < Math.max(n1, n2); ++i) {
      let i1 = i < n1 ? parseInt(nums1[i]) : 0;
      let i2 = i < n2 ? parseInt(nums2[i]) : 0;
      if (i1 != i2) {
        return i1 > i2 ? 1 : -1;
      }
     }

     return 0;
};



//166 https://leetcode.com/problems/fraction-to-recurring-decimal/description/


//Input: numerator = 1, denominator = 2
//Output: "0.5"



const fractionToDecimal = function (numerator, denominator) {
  if (numerator === 0) {
    return "0";
  }

  let fraction = [];
  if ((numerator < 0) ^ (denominator < 0)) {
    fraction.push("-");
  }

  let dividend = Math.abs(numerator);
  let divisor = Math.abs(denominator);
  fraction.push(Math.floor(dividend / divisor).toString());
  let remainder = dividend % divisor;
  if (remainder === 0) {
    return fraction.join("");
  }
  fraction.push(".");
  let map = {};
  while (remainder !== 0) {
    if (remainder in map) {
      fraction.splice(map[remainder], 0, "(");
      fraction.push(")");
      break;
    }
    map[remainder] = fraction.length;
    remainder *= 10;
    fraction.push(Math.floor(remainder / divisor).toString());
    remainder %= divisor;
  }
  return fraction.join("");
};



//167 https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/

// Input: numbers = [2,7,11,15], target = 9
//Output: [1,2]
//Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].


const twoSum = function (numbers, target) {
  let low = 0;
  let high = numbers.length - 1;
  while (low < high) {
    let sum = numbers[low] + numbers[high];

    if (sum == target) {
      return [low + 1, high + 1];
    } else if (sum < target) {
      low++;
    } else {
      high--;
    }
  }
  return [-1, -1];
};



// 168   Input: columnNumber = 1
//Output: "A"


const convertToTitle = function (columnNumber) {
  let ans = "";

  while (columnNumber > 0) {
    columnNumber--;
    ans = String.fromCharCode((columnNumber % 26) + "A".charCodeAt(0)) + ans;
    columnNumber = Math.floor(columnNumber / 26);
  }

  return ans;
};


//169 https://leetcode.com/problems/majority-element/


//Input: nums = [3,2,3]
//Output: 3


const majorityElement = function (nums) {
  let counts = {};

  for (let num of nums) {
    if (!counts[num]) {
      counts[num] = 1;
    } else {
      counts[num]++;
    }
  }

  for (let num in counts) {
    if (counts[num] > nums.length / 2) return Number(num);
  }
  return 0;
};


//170 https://leetcode.com/problems/two-sum-iii-data-structure-design/

//Input
//["TwoSum", "add", "add", "add", "find", "find"]
//[[], [1], [3], [5], [4], [7]]
//Output
//[null, null, null, null, true, false]


class TwoSum {
  constructor() {
    this.nums = [];
    this.is_sorted = false;
  }

  add(number) {
    this.nums.push(number);
    this.is_sorted = false;
  }

  find(value) {
    if (!this.is_sorted) {
      this.nums.sort((a, b) => a - b);
      this.is_sorted = true;
    }

    let low = 0,
    high = this.nums.length - 1;
    while (low < high) {
      let twosum = this.nums[low] + this.nums[high];
      if (twosum < value) low += 1;
      else if (twosum > value) high -= 1;
      else return true;
    }
    return false;
  }
}

//171 https://leetcode.com/problems/excel-sheet-column-number/

//Input: columnTitle = "A"
//Output: 1


const titleToNumber = function (s) {
  let result = 0;

  const alpha_map = {};
  for (let i = 0; i < 26; i++) {
    alpha_map[String.fromCharCode(i + 65)] = i + 1;
  }

  const n = s.length;
  for (let i = 0; i < n; i++) {
    let cur_char = s[n - 1 - i];
    result += alpha_map[cur_char] * Math.pow(26, i);
  }
  return result;
};


//172  Factorial Trailing Zeroes


//Input: n = 3
//Output: 0
//Explanation: 3! = 6, no trailing zero.


const trailingZeroes = function (n) {
  let nFactorial = BigInt(1);
  for (let i = 2; i <= n; i++) {
    nFactorial *= BigInt(i);
  }

  let zeroCount = 0;
  const ten = BigInt(10);

  while (nFactorial % ten === BigInt(0)) {
    nFactorial /= ten;
    zeroCount++;
  }

  return zeroCount;
};


//173 https://leetcode.com/problems/binary-search-tree-iterator/description/


//Input
//["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
//[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
//Output
//[null, 3, 7, true, 9, true, 15, true, 20, false]

//Explanation
//BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
//bSTIterator.next();    // return 3
//bSTIterator.next();    // return 7
//bSTIterator.hasNext(); // return True
//bSTIterator.next();    // return 9
//bSTIterator.hasNext(); // return True
//bSTIterator.next();    // return 15
//bSTIterator.hasNext(); // return True
//bSTIterator.next();    // return 20
//bSTIterator.hasNext(); // return False




class TreeNode {
  constructor(val) {
    this.val = val;
    this.left = null;
    this.right = null;
  }
}

class BSTIterator {
  constructor(root) {
    this.nodesSorted = [];
    this.index = -1;
    this._inorder(root);
  }

  _inorder(root) {
    if (root === null) {
      return;
    }

    this._inorder(root.left);
    this.nodesSorted.push(root.val);
    this._inorder(root.right);
  }

  next() {
    return this.nodesSorted[++this.index];
  }

  hasNext() {
    return this.index + 1 < this.nodesSorted.length;
  }
}


//174 https://leetcode.com/problems/dungeon-game/


//Input: dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
//Output: 7
//Explanation: The initial health of the knight must be at least 7 if he follows //the optimal path: RIGHT-> RIGHT -> DOWN -> DOWN.


const calculateMinimumHP = function (dungeon) {
  let rows = dungeon.length,
  cols = dungeon[0].length;
  let dp = Array(rows).fill()
  .map(() => Array(cols).fill(Number.MAX_SAFE_INTEGER));

  const get_min_health = (currCell, nextRow, nextCol) => {
    if (nextRow >= rows || nextCol >= cols) {
      return Number.MAX_SAFE_INTEGER;
    }
    let nextCell = dp[nextRow][nextCol];
    return Math.max(1, nextCell - currCell);
  };

  for (let row = rows - 1; row >= 0; --row) {
    for (let col = cols - 1; col >= 0; --col) {
      let currCell = dungeon[row][col];


      let right_health = get_min_health(currCell, row, col + 1);
      let down_health = get_min_health(currCell. row + 1, col);
      let next_health = Math.min(right_health, down_health);

      let min_health;
      if (next_health !== Number.MAX_SAFE_INTEGER) {

        min_health = next_health;
      } else {
        min_health = currCell >= 0 ? 1 : 1 - currCell;
      }

      dp[row][col] = min_health;
    }
  }
  return dp[0][0];
};



// 174 https://leetcode.com/problems/dungeon-game/description/


//Input: dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
//Output: 7
//Explanation: The initial health of the knight must be at least 7 if he follows the optimal path



class MyCircularQueue {
  constructor(capacity) {
    this.queue = new Array(capacity).fill(0);
    this.index = 0;
  }

  enQueue(value) {
    this.queue[this.index] = value;
    this.index = (this.index + 1) % this.queue.length;
  }

  get(index) {
    return this.queue[index % this.queue.length];
  }
}

function calculateMinimumHP(dungeon) {
  const rows = dungeon.length;
  const cols = dungeon[0].length;
  const dp = new MyCircularQueue(cols);
  const inf = Infinity;

  for (let row = rows - 1; row >= 0; --row) {
    for (let col = cols - 1; col >= 0; --col) {
      const currCell = dungeon[row][col];
      const rightHealth = col < cols - 1 ? Math.max(1, dp.get(col + 1) - currCell) : inf;

      const downHealth = row < rows - 1 ? Math.max(1, dp.get(col) - currCell) : inf;

      const minHealth = rightHealth === inf && downHealth === inf ? 1 : Math.min(rightHealth, downHealth);

      dp.enQueue(currCell >= 0 ? minHealth : minHealth - currCell);
    }
  }

  return dp.get(0);
};



//179 https://leetcode.com/problems/largest-number/

//Input: nums = [10,2]
//Output: "210"


class Solution {
  largestNumber(nums) {
    const strNums = nums.map(String);
    strNums.sort((a, b) => (b + a).localeCompare(a + b));

    if (strNums[0] === "0") {
      return "0";
    }
    return strNums.join('');
  }
}



//186 https://leetcode.com/problems/reverse-words-in-a-string-ii/


//Input: s = ["a"]
//Output: ["a"]


class Solution {
  reverseWords(s) {
    s.reverse();
    let start = 0, end = 0, n = s.length;


    while (start < n) {
      while (end < n && s[end] !== ' ') end++;
      s.slice(start, end).reverse().copyWithin(s, start);

      end++;
      start = end;
    }
  }
}



//187 https://leetcode.com/problems/repeated-dna-sequences/

//Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
//Output: ["AAAAACCCCC","CCCCCAAAAA"]


class Solution {
  largestNumber(nums) {
      const strNums = nums.map(String);
      strNums.sort((a, b) => (b + a).localeCompare(a + b));
      if (strNums[0] === "0") {
          return "0";
      }
      return strNums.join('');
  }
}


//188 https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/


//Input: k = 2, prices = [2,4,1]
//Output: 2
//Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.


class Solution {
  maxProfit(k, prices) {
    const n = prices.length;
    if (n === 0 || k === 0) return 0;
    if (k * 2 >= n) {
      let res = 0;
      for  (let i = 1; i < n; i++) {
        res += Math.max(0, prices[i] - prices[i - 1]);
      }
      return res;
    }

    const dp = Array.from({ length: n}, () => 
      Array.from({ length: k + 1}, () =>
        Array(2).fill(Number.MIN_SAFE_INTEGER / 2) 
      )
      
    );
    dp[0][0][0] = 0;
    dp[0][1][1] = -prices[0];

    for (let i = 1; i < n; i++) {
      for (let j = 0; j <= k; j++) {
        dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
        if (j > 0) {
          dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i -1][j -1][0] - prices[i]);
        }
      }
    }
    return Math.max(...dp[n - 1].map(x => x[0]));
  }
}


//189 https://leetcode.com/problems/rotate-array/


// Input: nums = [1,2,3,4,5,6,7], k = 3
//Output: [5,6,7,1,2,3,4]
//Explanation:
//rotate 1 steps to the right: [7,1,2,3,4,5,6]
//rotate 2 steps to the right: [6,7,1,2,3,4,5]
//rotate 3 steps to the right: [5,6,7,1,2,3,4]



class Solution {
  rotate(nums, k) {
    const n = nums.length;
    const a = new Array(n);

    for (let i = 0; i < n; i++) {
      a[(i + k) % n] = nums[i];
    }

    for (let i = 0; i < n; i++) {
      nums[i] = a[i];
    }
  }
}


//190 https://leetcode.com/problems/reverse-bits/

//Input: n = 00000010100101000001111010011100
//Output:    964176192 (00111001011110000010100101000000)
//Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.
//Example 2:


class Solution {
  constructor() {
    this.cache = {};
  }

  reverseByte(byte) {
    if (this.cache[byte] !== undefined) {
      return this.cache[byte];
    }
    let value = (byte * 0x0202020202 & 0x010884422010) % 1023;

    this.cache[byte] = value;
    return value;
  }

  reverseBits(n) {
    let ret = 0, power = 24;
    while (n !== 0) {
      ret += this.reverseByte(n & 0xff) << power;
      n = n >>> 8;
      power -= 8;
    }
    return ret;
  }
}


//191 https://leetcode.com/problems/number-of-1-bits/


//Input: n = 11

//Output: 3

//Explanation:

//The input binary string 1011 has a total of three set bits.


function hammingWeight(n) {
  let bits = 0;
  let mask = 1;
  for (let i = 0; i < 32; i++) {
    if ((n & mask) !== 0) {
      bits++;
    }
    mask <<= 1;
  }
  return bits;
}


//198 https://leetcode.com/problems/house-robber/


//Input: nums = [1,2,3,1]
//Output: 4
//Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
//Total amount you can rob = 1 + 3 = 4.


class Solution {
  constructor() {
    this.memo = [];
  }

  rob(nums) {
    this.memo = Array(100).fill(-1);
    return this.robFrom(0, nums);
  }

  robFrom(i, nums) {
    if (i >= nums.length) {
      return 0;
    }
    if (this.memo[i] !== -1) {
      return this.memo[i];
    }

    const ans = Math.max(this.robFrom(i + 1, nums), this.robFrom(i + 2, nums) + nums[i]);
    this.memo[i] = ans;
    return ans;
  }
}


//199 https://leetcode.com/problems/binary-tree-right-side-view/description/


//Input: root = [1,2,3,null,5,null,4]
//Output: [1,3,4]


class Solution {
  rightSideView(root) {
    if (root === null) return [];

    let nextLevel = [root];
    let rightside = [];

    while (nextLevel.length > 0) {
      let currLevel = nextLevel;
      nextLevel = [];


      for (let node of currLevel) {
        if (node.left !== null)
         nextLevel.push(node.left);
        if (node.right !== null)
          nextLevel.push(node.right); 
      }
      rightside.push(currLevel[currLevel.length - 1].val);
    }
    return rightside;
  }
}


// 200 Number of Islands


//Input: grid = [
  //["1","1","1","1","0"],
  //["1","1","0","1","0"],
  //["1","1","0","0","0"],
  //["0","0","0","0","0"]
//]
//Output: 1


class Solution {
  dfs(grid, r, c) {
    const nr = grid.length;
    const nc = grid[0].length;

    if (r < 0 || c < 0 || r >= nr || c >= nc || grid[r][c] === '0') {
      return;
    }

    grid[r][c] = '0';
    this.dfs(grid, r - 1, c);
    this.dfs(grid, r + 1, c);
    this.dfs(grid, r, c - 1);
    this.dfs(grid, r, c + 1);
  }

  numIslands(grid) {
    if (!grid || grid.length === 0) {
      return 0;
    }

    const nr = grid.length;
    const nc = grid[0].length;
    let numIslands = 0;

    for (let r = 0; r < nr; ++r) {
      for (let c = 0; c < nc; ++c) {
        if (grid[r][c] === '1') {
          numIslands++;
          this.dfs(grid, r, c);
        }
      }
    }

    return numIslands;
  }
}


//201 Bitwise AND of Numbers Range


//Input: left = 5, right = 7
//Output: 4


class Solution {
  rangeBitwiseAnd(m, n) {
    let shift = 0;
    while (m < n) {
      m >>= 1;
      n >>= 1;
      shift++;
    }
    return m << shift;
  }
}


//202 https://leetcode.com/problems/happy-number/description/

//Input: n = 19
//Output: true
//Explanation:
//12 + 92 = 82
//82 + 22 = 68
//62 + 82 = 100
//12 + 02 + 02 = 1



class Solution {
  isHappy(n) {
    function getNext(n) {
      let totalSum = 0;
      while (n > 0) {
        let digit = n % 10;
        totalSum += digit * digit;
        n = Math.floor(n / 10);
      }
      return totalSum;
    }

    let seen = new Set();
    while (n !== 1 && !seen.has(n)) {
      seen.add(n);
      n = getNext(n);
    }
    return n === 1;
  }
}


//202 https://leetcode.com/problems/happy-number/description/


//Input: n = 2
//Output: false


function getNext(n) {
  let totalSum = 0;
  while (n > 0) {
    let digit = n % 10;
    n = Math.floor(n / 10);
    totalSum += digit * digit;
  }
  return totalSum;
}

function isHappy(n) {
  const seen = new Set();
  while (n !== 1 && !seen.has(n)) {
    seen.add(n);
    n = getNext(n);
  }
  return n === 1;
}

console.log(isHappy(19));

//203 Remove Linked List Elements


//Input: head = [1,2,6,3,4,5,6], val = 6
//Output: [1,2,3,4,5]

function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

function deleteNode(head, value) {
  let sentinel = new ListNode(0);
  sentinel.next = head;
  let prev = sentinel, curr = head;

  while (curr !== null) {
    if (curr.val === value) {
      prev.next = curr.next;
    } else {
      prev = curr;
    }
    curr = curr.next;
  }
  return sentinel.next;
}


//204 https://leetcode.com/problems/count-primes/description/


//Input: n = 10
//Output: 4
//Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.


class Solution {
  countPrimes(n) {
    if (n <= 2) {
      return 0;
    }

    let numbers = new Array(n).fill(true);
    for (let p = 2; p <= Math.sqrt(n); ++p) {
      if (numbers[p]) {
        for (let j = p * p; j < n; j += p) {
          numbers[j] = false;
        }
      }
    }

    let numberOfPrimes = 0;
    for (let i = 2; i < n; i++) {
      if (numbers[i]) {
        ++numberOfPrimes;
      }
    }

    return numberOfPrimes;
  }
}


//205 Isomorphic Strings


// Input: s = "egg", t = "add"
//Output: true


class Solution {
  isIsomorphic(s, t) {
    let mappingStoT = {};
    let mappingTtoS = {};

    for (let i = 0; i < s.length; i++) {
      let c1 = s[i];
      let c2 = t[i];

      if (mappingStoT[c1] === undefined && mappingTtoS[c2] === undefined) {
        mappingStoT[c1] = c2;
        mappingTtoS[c2] = c1;
      
      } else if (mappingStoT[c1] !== c2 || mappingTtoS[c2] !== c1) {
        return false;
      }
    }
    return true;
  }
}


//206 https://leetcode.com/problems/reverse-linked-list/


// Input: head = [1,2,3,4,5]
//Output: [5,4,3,2,1]


function ListNode(val, next = null) {
  this.val = val;
  this.next = next;
}

class solution {
  reverselist(head) {
    let prev = null;
    let curr = head;
    while (curr !== null) {
      let nextTemp = curr.next;
      curr.next = prev;
      prev = curr;
      curr = nextTemp;
    }
    return prev;
  }
}










