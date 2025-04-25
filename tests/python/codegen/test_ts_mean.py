import tvm
from tvm import relay
import numpy as np
from tvm.relay.build_module import create_executor

def update_lib(lib):
    kwargs = {}
    kwargs["options"] = ["-O0", 
                         "-g3",
                         "-std=c++17",
                         "-I", "/home/duxiutao/repos/tvm/3rdparty/dlpack/include",
                         "-I", "/home/duxiutao/repos/tvm/include"]
    lib.export_library("myadd.so", fcompile=False, **kwargs, workspace_dir="/home/duxiutao/repos/tvm/build")
    return tvm.runtime.load_module("myadd.so")

dshape = (10,)
wshape = (1,)
mean_data_var = relay.var("mean_data_var", relay.TensorType(dshape, "float32"))
mean_window_var = relay.var("mean_window_var", relay.TensorType(wshape, "float32"))

mean_res = relay.my_ts_mean(mean_data_var, mean_window_var)

# 构建函数
func = relay.Function([mean_data_var, mean_window_var], mean_res)
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

device = tvm.cpu(0)
# executor = tvm.contrib.graph_executor.GraphModule(lib["default"](device))
try:
    aaa = lib["default"]
    print("default is OK: ", aaa)
except Exception as e:
    print("error is ", e)

executor = tvm.contrib.graph_executor.GraphModule(aaa(device))

mean_data = np.full(dshape, 11.0).astype("float32")
mean_window = np.full(wshape, 3.0).astype("int32")

i_window = tvm.nd.array(mean_window)
for v in range(5):
    mean_data = mean_data + 10**v;
    i_data = tvm.nd.array(mean_data)
    executor.set_input("mean_data_var", i_data)
    executor.set_input("mean_window_var", i_window)
    executor.run()
    print("=" * 80)
    output_data = executor.get_output(0).asnumpy()
    print("input:", mean_data[0], "output[", v, "]:", output_data)



