#include <vector>
#include <iostream>
using std::vector;


class Solution
{
public:
  void nextPermutation(std::vector<int>& nums) {
    if (nums.empty() || nums.size() == 1) return;
    long pivot = nums.size() - 1;
    while (pivot && nums[pivot - 1] >= nums[pivot])
      pivot--;

    if (!pivot) 
    {
      std::reverse(nums.begin(), nums.end());
      return;
    }
        
    long change = pivot;
    while (change != nums.size() - 1 && nums[change + 1] > nums[pivot - 1])
        change++;
    
    std::swap(nums[pivot - 1], nums[change]);
    std::reverse(pivot + nums.begin(), nums.end());
        
    for (long i = pivot; i < nums.size() - 1; i++)
    {
        if (nums[i] > nums[i + 1])
            std::swap(nums[i], nums[i + 1]);
    }
  }
}