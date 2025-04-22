import sys
import os
import numpy as np
import tvm
from tvm import relay
from tvm.relay.op import custom

# 添加自定义算子目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_network import CustomNetwork, test_custom_network

def main():
    print("Testing custom network with multiply_add operator...")
    test_custom_network()
    print("Test completed successfully!")

if __name__ == "__main__":
    main() 