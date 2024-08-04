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

