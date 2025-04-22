import tvm
from tvm import te
n = te.var("n")
A = te.placeholder((n, ), name = "A", dtype = "float32")
B = te.placeholder((n, ), name = "B", dtype = "float32")
C = te.compute(A.shape, lambda i: A[i] + B[i], name = "C")

s = te.create_schedule(C.op)

print(tvm.lower(s, [A, B, C], simple_mode = True))