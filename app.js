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
