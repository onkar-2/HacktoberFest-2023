#include <iostream>
using namespace std;

int main() {
    int arr[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int low = 0, high = (sizeof(arr) / sizeof(arr[0])) - 1;
    int num = 6;

    while (low <= high) {
        int mid = low + (high - low) / 2;  // To avoid overflow

        if (arr[mid] == num) {
            cout << "At: " << (mid + 1) << endl;
            break;
        } else if (arr[mid] < num) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return 0;
}
