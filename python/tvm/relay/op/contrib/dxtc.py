import tvm


@tvm.ir.register_op_attr("dxt_axis_abs", "target.dxtc")
def dxtc_support(expr):
    return True