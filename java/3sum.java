import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ThreeSum {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        
        // Check if the input array has fewer than 3 elements
        if (nums.length < 3) {
            return result; // Return empty result
        }
        
        Arrays.sort(nums); // Sort the array for two-pointer technique

        for (int i = 0; i < nums.length - 2; i++) {
            // Skip duplicates for the first number
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue; 
            }
            int target = -nums[i];
            int left = i + 1;
            int right = nums.length - 1;

            while (left < right) {
                int sum = nums[left] + nums[right];
                if (sum == target) {
                    // Found a triplet
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    left++;
                    right--;

                    // Skip duplicates for the second number
                    while (left < right && nums[left] == nums[left - 1]) {
                        left++;
                    }
                    // Fix for skipping duplicates for the third number
                    while (left < right && nums[right] == nums[right + 1]) {
                        right--;
                    }
                } else if (sum < target) {
                    left++; // Increase the left pointer
                } else {
                    right--; // Decrease the right pointer
                }
            }
        }

        return result; // Return the list of triplets
    }

    public static void main(String[] args) {
        ThreeSum solution = new ThreeSum();
        int[] nums = {-1, 0, 1, 2, -1, -4};
        List<List<Integer>> result = solution.threeSum(nums);

        // Print the results
        for (List<Integer> triplet : result) {
            System.out.println(triplet);
        }
    }
}
