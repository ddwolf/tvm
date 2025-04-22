#ifndef TVM_RELAY_OP_CUSTOM_CUSTOM_MULTIPLY_ADD_H_
#define TVM_RELAY_OP_CUSTOM_CUSTOM_MULTIPLY_ADD_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

Expr MakeCustomMultiplyAdd(Expr data1, Expr data2, Expr data3);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_CUSTOM_CUSTOM_MULTIPLY_ADD_H_ 