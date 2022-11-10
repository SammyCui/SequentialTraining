def solution(a):
    largest_num_in_input = max(a)
    all_fib_nums = [1, 1]
    while all_fib_nums[-1] < largest_num_in_input:
        all_fib_nums.append(all_fib_nums[-1] + all_fib_nums[-2])

    def check_sum(x):
        for fib in all_fib_nums:
            if x - fib in all_fib_nums:
                return True
        return False

    return [check_sum(x) for x in a]

if __name__ == '__main__':
    print(solution([1,2,3,6]))