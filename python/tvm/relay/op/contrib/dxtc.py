import tvm


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.custom")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper

_register_external_op_helper("dxt_axis_abs")
_register_external_op_helper("my_ts_mean")
