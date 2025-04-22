import tvm
from tvm import relay
from tvm.relay import testing
import numpy as np

def test_custom_multiply_add():
    # 创建输入张量
    shape = (10,)
    dtype = "float32"
    a = relay.var("a", shape=shape, dtype=dtype)
    b = relay.var("b", shape=shape, dtype=dtype)
    c = relay.var("c", shape=shape, dtype=dtype)
    
    # 创建自定义操作
    custom_op = relay.op.get("custom_multiply_add")
    y = relay.Call(custom_op, [a, b, c])
    
    # 创建函数
    func = relay.Function([a, b, c], y)
    
    # 创建输入数据
    np_a = np.random.uniform(size=shape).astype(dtype)
    np_b = np.random.uniform(size=shape).astype(dtype)
    np_c = np.random.uniform(size=shape).astype(dtype)
    
    # 编译和运行
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
    
    # 创建运行时
    ctx = tvm.cpu()
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx))
    
    # 设置输入
    runtime.set_input("a", np_a)
    runtime.set_input("b", np_b)
    runtime.set_input("c", np_c)
    
    # 运行
    runtime.run()
    
    # 获取输出
    out = runtime.get_output(0).numpy()
    
    # 验证结果
    expected = np_a * np_b + np_c
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    
    print("Test passed!")

if __name__ == "__main__":
    test_custom_multiply_add() 