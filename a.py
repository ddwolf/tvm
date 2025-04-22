import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine
import numpy as np
from tvm import te
from tvm.ir.op import Op

# 1. 定义并注册一个外部函数 (PackedFunc)
def my_add_numpy(x, y, out):
    np.add(x, y, out=out)

f_my_add = tvm.register_func(my_add_numpy, override=True)

# 2. 在 Relay 中定义一个 extern op
op_name = "my_extern_add"

@relay.op.register(op_name)
class MyExternAddOp(relay.op.OpPattern):
    num_inputs = 2
    support_level = 10

    @staticmethod
    def compute(attrs, args, targs):
        # 使用 te.extern 调用我们的 PackedFunc
        return te.extern(
            args[0].shape,
            [args[0], args[1]],
            lambda ins, outs: tvm.nd.NDArray.copyto(f_my_add(ins[0], ins[1]), outs[0]),
            dtype=targs[0].dtype,
            name="extern_op_compute"
        )

    @staticmethod
    def infer_type(type_infer, orig_expr):
        a_type = orig_expr.args[0].checked_type
        b_type = orig_expr.args[1].checked_type
        if a_type != b_type:
            raise TypeError("Input types must be the same")
        return a_type

my_extern_add = Op.get(op_name)

# 3. 创建一个使用该 extern op 的 Relay 程序
def create_relay_program():
    a = relay.var("a", relay.TensorType((2, 2), "float32"))
    b = relay.var("b", relay.TensorType((2, 2), "float32"))
    output = my_extern_add(a, b)
    return relay.Function([a, b], output)

relay_prog = create_relay_program()

# 4. 构建 Relay 程序
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(relay_prog, target=target, params={})

# 5. 创建一个 Relay VM 并运行程序
ex = relay.vm.compile(relay_prog, target=target, params={})
vm = VirtualMachine(ex, tvm.cpu())

# 6. 提供输入数据
input_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
input_b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

# 7. 运行程序
result = vm.invoke("main", tvm.nd.array(input_a), tvm.nd.array(input_b))
print("Result:")
print(result)
