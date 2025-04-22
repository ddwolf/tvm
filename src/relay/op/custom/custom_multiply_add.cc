#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <vector>

namespace tvm {
namespace relay {

// 定义算子的属性结构体
struct CustomMultiplyAddAttrs : public tvm::AttrsNode<CustomMultiplyAddAttrs> {
  TVM_DECLARE_ATTRS(CustomMultiplyAddAttrs, "relay.attrs.CustomMultiplyAddAttrs") {
  }
};

// 注册属性结构体
TVM_REGISTER_NODE_TYPE(CustomMultiplyAddAttrs);

// 定义类型关系函数
bool CustomMultiplyAddRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                         const TypeReporter& reporter) {
  // 检查输入参数数量
  ICHECK_EQ(types.size(), 4);
  const auto* data1 = types[0].as<TensorTypeNode>();
  const auto* data2 = types[1].as<TensorTypeNode>();
  const auto* data3 = types[2].as<TensorTypeNode>();
  
  if (data1 == nullptr || data2 == nullptr || data3 == nullptr) {
    return false;
  }

  // 检查输入张量的形状是否相同
  ICHECK(data1->shape.size() == data2->shape.size() && data1->shape.size() == data3->shape.size())
      << "CustomMultiplyAdd: shapes of inputs must be the same";

  // 设置输出类型
  reporter->Assign(types[3], TensorType(data1->shape, data1->dtype));
  return true;
}

// 定义算子的计算函数
Expr MakeCustomMultiplyAdd(Expr data1, Expr data2, Expr data3) {
  auto attrs = make_object<CustomMultiplyAddAttrs>();
  static const Op& op = Op::Get("custom.multiply_add");
  return Call(op, {data1, data2, data3}, Attrs(attrs), {});
}

// 注册算子
TVM_REGISTER_GLOBAL("relay.op.custom._make.multiply_add")
    .set_body_typed(MakeCustomMultiplyAdd);

// 注册算子的属性
RELAY_REGISTER_OP("custom.multiply_add")
    .describe(R"code(Element-wise multiply-add operation.
    
    .. math::
        out = data1 * data2 + data3

    )code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("data1", "Tensor", "First input tensor")
    .add_argument("data2", "Tensor", "Second input tensor")
    .add_argument("data3", "Tensor", "Third input tensor")
    .set_support_level(1)
    .add_type_rel("CustomMultiplyAdd", CustomMultiplyAddRel);

}  // namespace relay
}  // namespace tvm 