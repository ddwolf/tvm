#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <sstream>

namespace tvm {
namespace relay {
namespace contrib {

class DXTAxisAbsCodegen : public ExprVisitor {
 public:
  explicit DXTAxisAbsCodegen(const std::string& id) { this->ext_func_id_ = id; }

  void VisitExpr_(const CallNode* call) final {
    ExprVisitor::VisitExpr_(call);
    if (call->op.as<OpNode>()) {
      const auto* op = call->op.as<OpNode>();
      if (op->name == "custom_op") {
        // 获取输入张量
        auto input1 = call->args[0];
        auto input2 = call->args[1];
        
        // 获取属性
        const auto* attrs = call->attrs.as<DXTAxisAbsAttrs>();
        int axis = attrs->axis;
        int indice = attrs->indice;
        
        // 生成C代码
        std::stringstream code;
        code << "void custom_op_" << ext_func_id_ << "("
             << "float* input1, float* output, "
             << "int axis, int indice, "
             << "int size) {\n"
             << "  for (int i = 0; i < size; ++i) {\n"
             << "    output[i] = input1[i] * param2 + input2[i] * param1;\n"
             << "  }\n"
             << "}\n";
        
        this->code_ = code.str();
      }
    }
  }

  std::string GetCode() { return this->code_; }

 private:
  std::string ext_func_id_;
  std::string code_;
};

runtime::Module DXTAxisAbsCompiler(const ObjectRef& ref) {
  // 创建代码生成器
  DXTAxisAbsCodegen codegen("dxt_axis_abs");
  
  // 遍历Relay表达式
  if (ref->IsInstance<FunctionNode>()) {
    codegen.VisitExpr(Downcast<Function>(ref));
  } else {
    LOG(FATAL) << "The input ref is expected to be a Relay function or module";
  }

  // 生成C代码
  std::string code = codegen.GetCode();
  
  // 创建运行时模块
  const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
  ICHECK(pf != nullptr) << "Cannot find CSource module to create the external runtime module";
  return (*pf)(code, "c", Array<String>{});
}

TVM_REGISTER_GLOBAL("relay.ext.custom").set_body_typed(DXTAxisAbsCompiler);
TVM_REGISTER_GLOBAL("target.build.custom")
    .set_body_typed([](IRModule mod, Target target) -> runtime::Module {
      return DXTAxisAbsCompiler(mod);
    });
}  // namespace contrib
}  // namespace relay
}  // namespace tvm 