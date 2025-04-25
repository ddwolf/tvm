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
    lib.export_library("mymulti.so", fcompile=False, **kwargs, workspace_dir="/home/duxiutao/repos/tvm/build")
    return tvm.runtime.load_module("mymulti.so")

dshape = (1,)
ori = relay.var("ori", relay.TensorType(dshape, "float32"))
win = relay.var("win", relay.TensorType((1,), "int32"))
mean = relay.my_ts_mean(ori, win)
op2_var = relay.var("op2", relay.TensorType(dshape, "float32"))
pdt = relay.my_multi(mean, op2_var)
func = relay.Function([ori, win, op2_var], pdt)
mod = tvm.IRModule({"main": func})
target = tvm.target.Target("custom", host="llvm")
print("=" * 80)
# 构建
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=None)

lib = update_lib(lib)

device = tvm.cpu(0)
try:
    aaa = lib["default"]
    print("default is OK: ", aaa)
except Exception as e:
    print("error is ", e)

executor = tvm.contrib.graph_executor.GraphModule(aaa(device))

ori_data = np.full(dshape, 11.0).astype("float32")
win_data = np.full((1,), 3).astype("int32")
op2_data = np.full(dshape, 3.0).astype("float32")

orid = tvm.nd.array(ori_data)
wind = tvm.nd.array(win_data)
op2d = tvm.nd.array(op2_data)
for i in range(5):

    executor.set_input("ori", orid)
    print("Wind is ", wind, "ori is", orid)
    executor.set_input("win", wind)
    executor.set_input("op2", op2d)
    executor.run()
    ori_data = ori_data + 5;
    orid = tvm.nd.array(ori_data)
    print("===={}: output is {}".format(i, executor.get_output(0).asnumpy()))

print("=" * 80)
output_data = executor.get_output(0).asnumpy()
print("input:", ori_data[0], "", op2_data[0], "output[0]:", output_data[0])

print("final output: ", output_data)