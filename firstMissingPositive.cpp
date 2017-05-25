#include <vector>
#include <iostream>
using std::vector;

void print_vec(const std::vector<int>& vec)
{
	for (const auto& num : vec)
		std::cout << num << ' ';

	std::cout << std::endl;
}

class Solution {
public:
	int firstMissingPositive(vector<int>& nums)
	{
		if (nums.empty()) return 1;
		for (long i = 0; i < nums.size(); i++)
		{
			while (nums[i] != i + 1 && nums[nums[i] - 1] != nums[i] 
				&& nums[i] >= 1 && nums[i] <= nums.size())
			{
				std::swap(nums[i], nums[nums[i] - 1]);
				print_vec(nums);
			}
		}

		long i;
		for (i = 0; i < nums.size(); i++)
		{
			if (nums[i] != i + 1)
				break;
		}

		return i + 1;
	}
};

int main ()
{
	vector<int> case1 = {3, 4, -1, 1};
	Solution s;
	std::cout << s.firstMissingPositive(case1) << std::endl;
	return 0;
}