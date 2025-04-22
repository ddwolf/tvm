import tvm
from tvm import relay
from tvm.relay import nn
import numpy as np

class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.conv2d(
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW"
        )
        self.conv2 = nn.conv2d(
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW"
        )
        self.dense = nn.dense(units=10)

    def forward(self, x):
        # 第一个卷积层
        x = self.conv1(x)
        x = relay.nn.relu(x)
        x = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2))
        
        # 第二个卷积层
        x = self.conv2(x)
        x = relay.nn.relu(x)
        x = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2))
        
        # 展平
        x = relay.nn.batch_flatten(x)
        
        # 使用自定义算子
        # 创建三个相同形状的张量用于测试自定义算子
        shape = relay.shape_of(x)
        a = relay.var("a", shape=shape, dtype="float32")
        b = relay.var("b", shape=shape, dtype="float32")
        c = relay.var("c", shape=shape, dtype="float32")
        
        # 使用自定义算子
        x = relay.op.custom.multiply_add(x, a, b, c)
        
        # 全连接层
        x = self.dense(x)
        return x

def test_custom_network():
    # 创建网络实例
    net = CustomNetwork()
    
    # 创建输入数据
    batch_size = 1
    input_shape = (batch_size, 3, 32, 32)
    input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
    
    # 创建用于自定义算子的输入
    a_data = np.random.uniform(-1, 1, size=(batch_size, 64*8*8)).astype("float32")
    b_data = np.random.uniform(-1, 1, size=(batch_size, 64*8*8)).astype("float32")
    c_data = np.random.uniform(-1, 1, size=(batch_size, 64*8*8)).astype("float32")
    
    # 创建输入变量
    data = relay.var("data", shape=input_shape, dtype="float32")
    a = relay.var("a", shape=(batch_size, 64*8*8), dtype="float32")
    b = relay.var("b", shape=(batch_size, 64*8*8), dtype="float32")
    c = relay.var("c", shape=(batch_size, 64*8*8), dtype="float32")
    
    # 构建网络
    net_out = net(data)
    
    # 创建函数
    func = relay.Function([data, a, b, c], net_out)
    
    # 编译
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
    
    # 创建运行时
    ctx = tvm.cpu()
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx))
    
    # 设置输入
    module.set_input("data", input_data)
    module.set_input("a", a_data)
    module.set_input("b", b_data)
    module.set_input("c", c_data)
    
    # 运行
    module.run()
    
    # 获取输出
    output = module.get_output(0)
    print("Output shape:", output.shape)
    print("Output sample:", output.numpy()[0, :5])

if __name__ == "__main__":
    test_custom_network() 