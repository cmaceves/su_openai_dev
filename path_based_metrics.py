import os
import sys
import json


def main():
    filename = "/home/caceves/su_openai_dev/no_follow/Clofedanol_Cough.json"
    with open(filename, 'r') as rfile:
        data = json.load(rfile)
    print(data.keys())


if __name__ == "__main__":
    main()
