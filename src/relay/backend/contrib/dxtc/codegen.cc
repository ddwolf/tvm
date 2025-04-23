#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/stmt.h>

#include <fstream>
#include <sstream>

namespace tvm {
namespace relay {
namespace contrib {
class PrimFuncCodeGen : public tvm::tir::StmtVisitor {
 public:
  explicit PrimFuncCodeGen(std::ostream& os) : os_(os), indent_(0) {}

  void GenerateCode(const tvm::tir::PrimFunc& func) {
    auto fnname = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(fnname.defined()) << " fn name undefined";
    auto strname = fnname.value();
    os_ << "// CodeGen for function: " << strname << "\n";
    const auto *op = func.operator->();
    const auto &signature = op->func_type_annotation();
    os_ << "// Signature: " << signature << "\n";
    os_ << "#include <stdio.h>\n";
    os_ << "#include <tvm/te/tensor.h>\n";
    os_ << "/** see C_backend_api.h#TVMBackendPackedCFunc\n"
        << " * \\param TVMValue *args The arguments\n"
        << " * \\param int *type_codes The type codes of the arguments\n"
        << " * \\param int num_args Number of arguments\n"
        << " * \\param TVMValue *out_ret_value The output value of the return value\n"
        << " * \\param int *out_ret_tcode The output type code of the return value\n"
        << " * \\param void *resource_handle Pointer to associated resource\n"
        << " */"
        << " extern \"C\" int " << strname << "(\n"
        << "        void *args,\n" 
        << "        int *type_codes,\n"
        << "        int num_args,\n"
        << "        void *out_ret_value,\n"
        << "        int *out_ret_tcode,\n"
        << "        void *resource_handle) {\n"
        << "  printf(\"args=%p, type_codes=%p, num_args=%d, out_ret_value=%p, out_ret_tcode=%p, resource_handle=%p\",\n"
        << "         args, type_codes, num_args, out_ret_value, out_ret_tcode, resource_handle); \n"
        << "  for (int i = 0; i < 27; ++i) {\n"
        << "    //((int*)output)[i] = -(*((int*)data + i));\n"
        << "  }\n"
        << "  return 0;\n"
        << "}\n";
    //VisitStmt(func->body);
  }

 protected:
  std::ostream& os_;
  int indent_;

  void PrintIndent() {
    for (int i = 0; i < indent_; ++i) os_ << "  ";
  }

  void VisitStmt_(const tvm::tir::ForNode* op) override {
    PrintIndent();
    os_ << "for (int " << op->loop_var->name_hint
        << " = " << PrintExpr(op->min) << "; "
        << op->loop_var->name_hint << " < " << PrintExpr(op->min) << " + " << PrintExpr(op->extent) << "; "
        << "++" << op->loop_var->name_hint << ") {\n";
    indent_++;
    VisitStmt(op->body);
    indent_--;
    PrintIndent();
    os_ << "}\n";
  }

  void VisitStmt_(const tvm::tir::LetStmtNode* op) override {
    PrintIndent();
    os_ << "auto " << op->var->name_hint << " = " << PrintExpr(op->value) << ";\n";
    VisitStmt(op->body);
  }

  void VisitStmt_(const tvm::tir::EvaluateNode* op) override {
    PrintIndent();
    os_ << PrintExpr(op->value) << ";\n";
  }

  void VisitStmt_(const tvm::tir::SeqStmtNode* op) override {
    for (const auto& stmt : op->seq) {
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const tvm::tir::BufferStoreNode* op) override {
    PrintIndent();
    os_ << op->buffer->name << "[" << PrintExpr(op->indices[0]) << "] = " << PrintExpr(op->value) << ";\n";
  }

  void VisitStmt_(const tvm::tir::IfThenElseNode* op) override {
    PrintIndent();
    os_ << "if (" << PrintExpr(op->condition) << ") {\n";
    indent_++;
    VisitStmt(op->then_case);
    indent_--;
    if (op->else_case.defined()) {
      PrintIndent();
      os_ << "} else {\n";
      indent_++;
      VisitStmt(op->else_case.value());
      indent_--;
    }
    PrintIndent();
    os_ << "}\n";
  }

  std::string PrintExpr(const PrimExpr& expr) {
    std::ostringstream oss;
    // 这里你可以扩展使用 ExprVisitor 或者使用 tvm::tir::ExprPrinter
    oss << expr;
    return oss.str();
  }
};

runtime::Module DXTAxisAbsCompiler(const ObjectRef& ref) {
  // 创建代码生成器
  std::ostringstream _oss;
  PrimFuncCodeGen codegen(_oss);
  
  // 遍历Relay表达式
  if (ref->IsInstance<FunctionNode>()) {
    std::cout << "What the ?" << std::endl;
  } else {
    auto x = ref.as<tvm::IRModule>();
    if (x != nullptr) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        codegen.GenerateCode(Downcast<tvm::tir::PrimFunc>(it.second));
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module";
    }
  }

  // 生成C代码
  std::string code = _oss.str()  ;
  std::cout << "CODES ARE:\n" << code << std::endl;

  // 创建运行时模块
  const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
  ICHECK(pf != nullptr) << "Cannot find CSource module to create the external runtime module";
  return (*pf)(code, "c", Array<String>{}, Array<String>{});
}

TVM_REGISTER_GLOBAL("relay.ext.custom").set_body_typed(DXTAxisAbsCompiler);
TVM_REGISTER_GLOBAL("target.build.custom")
    .set_body_typed([](IRModule mod, Target target) -> runtime::Module {
      return DXTAxisAbsCompiler(mod);
    });
}  // namespace contrib
}  // namespace relay
}  // namespace tvm 