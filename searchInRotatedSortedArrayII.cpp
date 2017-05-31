#include <vector>
#include <iostream>
using std::vector;

class Solution {
public:
  bool search(const vector<int>& nums, int target) {
    if (nums.empty()) return false;

    int low = 0, high = nums.size() - 1, mid(0);
    while (low <= high)
    {
      int mid = low + (high - low) / 2;
      if (nums[low] == target || nums[mid] == target || nums[high] == target)
        return true;

      while (nums[low] == nums[mid] && low <= mid) low++;
      while (nums[high] == nums[mid] && high >= mid) high--;

      mid = low + (high - low) / 2;

      if (nums[low] < nums[mid] && nums[mid] < nums[high])
      {
        nums[mid] > target ? high = mid - 1 : low = mid + 1;
      }
      else if (nums[low] > nums[mid] && nums[mid] > nums[high])
      {
        nums[mid] > target ? low = mid + 1 : high = mid - 1;
      }
      else if (nums[low] > nums[mid] && nums[mid] < nums[high])
      {
        // 5,1,2,3,4
        nums[high] > target && nums[mid] < target ? low = mid + 1 : high = mid - 1;
      }
      else if (nums[low] < nums[mid] && nums[mid] > nums[high])
      {
        // 4,5,1,2,3
        nums[low] < target && nums[mid] > target ? high = mid - 1 : low = mid + 1;
      }
    }

    return false;
  }
};

int main ()
{
  vector<int> case1 = {4,5,1,2,3};
  vector<int> case2 = {5,1,2,3,4};
  vector<int> case3 = {1,1,3,1};
}