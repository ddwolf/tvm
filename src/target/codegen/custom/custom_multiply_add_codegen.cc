#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/ir/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op_attr_types.h>

#include <string>
#include <vector>
#include <sstream>

namespace tvm {
namespace codegen {

using namespace tir;

// 注册 call_extern 操作
TVM_REGISTER_OP("call_extern")
    .set_num_inputs(-1)
    .add_argument("func_name", "String", "Function name")
    .add_argument("args", "Array<Expr>", "Function arguments")
    .set_support_level(10);

class CustomMultiplyAddCodeGen : public ExprFunctor<void(const PrimExpr&, std::ostream&)> {
 public:
  void VisitExpr_(const CallNode* op, std::ostream& os) override {
    if (op->op.same_as(builtin::call_extern())) {
      auto func_name = Downcast<StringImm>(op->args[0])->value;
      if (func_name == "custom_multiply_add") {
        os << "custom_multiply_add(";
        for (size_t i = 1; i < op->args.size(); ++i) {
          if (i > 1) os << ", ";
          this->VisitExpr(op->args[i], os);
        }
        os << ")";
      } else {
        ExprFunctor::VisitExpr_(op, os);
      }
    } else {
      ExprFunctor::VisitExpr_(op, os);
    }
  }

  void VisitExpr_(const VarNode* op, std::ostream& os) override {
    os << op->name_hint;
  }

  void VisitExpr_(const IntImmNode* op, std::ostream& os) override {
    os << op->value;
  }

  void VisitExpr_(const FloatImmNode* op, std::ostream& os) override {
    os << op->value;
  }
};

// 注册自定义操作
TVM_REGISTER_OP("custom_multiply_add")
    .set_num_inputs(3)
    .add_argument("a", "Tensor", "First input tensor")
    .add_argument("b", "Tensor", "Second input tensor")
    .add_argument("c", "Tensor", "Third input tensor")
    .set_support_level(10);

// 注册代码生成器
TVM_REGISTER_GLOBAL("relay.ext.custom_multiply_add")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      // 生成C代码
      std::ostringstream code;
      code << "#include <tvm/runtime/c_runtime_api.h>\n";
      code << "#include <tvm/runtime/c_backend_api.h>\n";
      code << "#include <stdio.h>\n";
      code << "#include <assert.h>\n\n";
      
      code << "void custom_multiply_add(float* a, float* b, float* c, float* out, int size) {\n";
      code << "  // 参数检查\n";
      code << "  assert(a != NULL && b != NULL && c != NULL && out != NULL);\n";
      code << "  assert(size > 0);\n\n";
      
      code << "  // 执行计算\n";
      code << "  for (int i = 0; i < size; ++i) {\n";
      code << "    out[i] = a[i] * b[i] + c[i];\n";
      code << "  }\n";
      code << "  printf(\"Custom multiply-add operation completed successfully!\\n\");\n";
      code << "}\n";
      
      // 返回生成的代码
      *rv = code.str();
    });

}  // namespace codegen
}  // namespace tvm 