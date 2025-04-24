import tvm
from tvm import relay
import numpy as np
from tvm.relay.build_module import create_executor

def update_lib(lib):
    kwargs = {}
    kwargs["options"] = ["-O0", "-g3", "-std=c++17"]
    lib.export_library("myadd.so", fcompile=False, **kwargs)
    return tvm.runtime.load_module("myadd.so")

dshape = (3,3,3)
axis = 1
indice = 1
# 创建自定义算子调用
x = relay.var("x", relay.TensorType(dshape, "int32"))  # 定义relay输入tensor
y = relay.dxt_axis_abs(x, axis=1, indice=1)    # 定义axis_abs运算表达式

# 构建函数
func = relay.Function([x], y)
mod = tvm.IRModule({"main": func})
print("mod is ", mod)

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
    lib = relay.build(mod, target=target, params=None)

lib = update_lib(lib)
# myabs = new_lib["tvmgen_default_fused_dxt_axis_abs"]
# myabs2 = new_lib.get_function("tvmgen_default_fused_dxt_axis_abs", True)
# myabs3 = lib.lib["tvmgen_default_fused_dxt_axis_abs"]
# myabs4 = lib.lib.get_function("tvmgen_default_fused_dxt_axis_abs", True)

# inputdata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
# outputdata = [0] * 27

# ret = myabs3(inputdata,  (3,3,3), 3, None, None, outputdata)
# ret = myabs4(inputdata,  (3,3,3), 3, None, None, outputdata)
# ret = myabs2(inputdata,  (3,3,3), 3, None, None, outputdata)
# ret = myabs(inputdata,  (3,3,3), 3, None, None, outputdata)
# print("ret is ", ret)
##############################################################################################################

data = np.full(dshape, -1).astype("int32")
print("data is ", data)
device = tvm.cpu(0)
# executor = tvm.contrib.graph_executor.GraphModule(lib["default"](device))
try:
    aaa = lib["default"]
    print("default is OK: ", aaa)
except Exception as e:
    print("error is ", e)

# try:
#     aaa = lib["main"]
#     print("main is OK: ", aaa)
# except Exception as e:
#     print("error is ", e)

executor = tvm.contrib.graph_executor.GraphModule(aaa(device))

x_data = np.full(dshape, -1).astype("int32")
i_data = tvm.nd.array(x_data)

executor.set_input("x", i_data)
executor.run()

output_data = executor.get_output(0).asnumpy()

print("=" * 80)
print("output is ", output_data)
