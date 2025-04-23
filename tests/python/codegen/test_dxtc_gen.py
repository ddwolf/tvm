import tvm
from tvm import relay
import numpy as np
from tvm.relay.build_module import create_executor

dshape = (3,3,3)
axis = 1
indice = 1
# 创建自定义算子调用
x = relay.var("x", relay.TensorType(dshape, "int32"))  # 定义relay输入tensor
y = relay.dxt_axis_abs(x, axis=1, indice=1)    # 定义axis_abs运算表达式

# 构建函数
func = relay.Function([x], y)

# 设置目标为自定义代码生成器
target = tvm.target.Target("custom", host="llvm")

# with tvm.transform.PassContext(opt_level=3) as ctx:
#     graph_json, lib, params = relay.build_module.build(func, target="llvm", mod_name="main")
# print("graph json", graph_json, "lib", lib, "params", params)

# lowered = relay.build_module._build_for_device(func, target="llvm")
# print(lowered["main"])
print("=" * 80)
# 构建
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target=target, params=None)

lib.lib.export_library("myadd.so")
##############################################################################################################

data = np.full(dshape, -1).astype("int32")
device = tvm.cpu(0)
executor = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

x_data = np.full(dshape, -1).astype("int32")
i_data = tvm.nd.array(x_data)

executor.set_input("x", i_data)
executor.run()

output_data = executor.get_output(0).asnumpy()

print("=" * 80)
print("output is ", output_data)
