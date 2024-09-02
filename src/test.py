import os
from group.group import *

test_path = "./src/tests/"


def execute_test():
    print("Running tests...", os.listdir(os.curdir))
    for test in os.listdir(test_path):
        if test.endswith(".py"):
            print("\033[93m" + f"Running test: {test}" + "\033[0m")
            os.system(f"python {test_path}{test}")
            print()
    print("\033[92m" + "✅ All tests ran successfully!" + "\033[0m")


if __name__ == "__main__":
    execute_test()
