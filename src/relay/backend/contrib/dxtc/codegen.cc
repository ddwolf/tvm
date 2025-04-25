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

std::string GetFnName(const tvm::tir::PrimFunc& func) {
  auto fnname = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(fnname.defined()) << " fn name undefined";
  auto strname = fnname.value();
  return strname;
}

class PrimFuncCodeGen : public tvm::tir::StmtVisitor {
 public:
  explicit PrimFuncCodeGen(std::ostream& os) : os_(os), indent_(0) {}

  void GenerateCode(const tvm::tir::PrimFunc& func) {
    std::string fnname = GetFnName(func);
    if (fnname.find("dxt_axis_abs") != std::string::npos) {
      GenerateCodeAxisAbs(func, fnname);
    } else if (fnname.find("my_ts_mean") != std::string::npos) {
      GenerateCodeTsMean(func, fnname);
    } else if (fnname.find("my_multi") != std::string::npos) {
      GenerateCodeMulti(func, fnname);
    } else {
      LOG(FATAL) << "Unsupported function: " << fnname;
    }
  }
  void GenerateCodeMulti(const tvm::tir::PrimFunc& func, const std::string& fnname) {
    os_ << "// CodeGen for function: " << fnname << "\n";
    const auto *op = func.operator->();
    const auto &signature = op->func_type_annotation();
    os_ << "// Signature: " << signature << "\n";
    os_ << R"CPP(
#include <iostream>
#include <cmath>
#include <cassert>
#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
using namespace std;
/**
 * \param TVMValue *args The arguments
 * \param int *type_codes The type codes of the arguments
 * \param int num_args Number of arguments
 * \param TVMValue *out_ret_value The output value of the return value
 * \param int *out_ret_tcode The output type code of the return value
 * \param void *resource_handle Pointer to associated resource
 */
)CPP";
    os_ << "extern \"C\" int " << fnname << "(\n";
    os_ << R"CPP(
        void *args,
        int *type_codes,
        int num_args,
        void *out_ret_value,
        int *out_ret_tcode,
        void *resource_handle) {

    DLTensor *op1 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
    float *op1data = (float*)op1->data;
    DLTensor *op2 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
    assert(op2->ndim == op1->ndim);
    assert(op2->ndim == 1);
    float *op2data = (float*)op2->data;
    DLTensor *output = (DLTensor*)(((TVMValue*)args)[2].v_handle);
    float *outdata = (float*)output->data;
    const int datalen = op1->shape[0];
    assert(datalen > 0);
    for (int i = 0; i < datalen; ++i) {
      outdata[i] = op1data[i] * op2data[i];
    }
    printf("args=%p, type_codes=%p, num_args=%d, out_ret_value=%p, out_ret_tcode=%p, resource_handle=%p\n",
           args, type_codes, num_args, out_ret_value, out_ret_tcode, resource_handle);
  for (int n = 0; n < num_args; ++n) {
    printf("codegen: === type[%d]=%d, args[%d]=%p\n", n, type_codes[n], n, &((TVMValue*)args)[n]);
    DLTensor *data = (DLTensor*)(((TVMValue*)args)[n].v_handle);
    printf("codegen: data=%p, data->data=%p, data->ndim=%d\n", data, data->data, data->ndim);
    float *outdata = (float*)data->data;
    printf("codegen: outdata=%p, outdata[0]=%f, ndim=%d\n", outdata, outdata[0], data->ndim);
    for (int m = 0; m < data->ndim; ++m) {
      printf("codegen: - shape[%d]=%ld\n", m, data->shape[m]);
    }
  }
  return 0;
}
        )CPP";
  }
  void GenerateCodeTsMean(const tvm::tir::PrimFunc& func, const std::string& fnname) {
  os_ << "// CodeGen for function: " << fnname << "\n";
    const auto *op = func.operator->();
    const auto &signature = op->func_type_annotation();
    os_ << "// Signature: " << signature << "\n";
    os_ << R"(#include <iostream>
#include <iostream>
#include <cmath>
#include <cassert>
#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
using namespace std;

template<typename T>
class TS_MEAN_CY {
public:
  TS_MEAN_CY() = default;
  void init(int window) {
    this->window = window;
    this->step = 0;
    this->index = 0;
    this->data = new T[window];
  }
  ~TS_MEAN_CY() { delete[] data; }
  T rolling(const T &new_value) {
    if (std::isnan(new_value)) {
      nan_count++;
    } else{       sum = sum + new_value;
    }
    step++;
    if (std::isnan(data[index])) {
      nan_count--;
    } else {
      sum = sum - data[index];
    }
    data[index] = new_value;
    index = (index + 1) % window;
    if (window == nan_count) {
      return new_value;
    } else {
      return sum / (window - nan_count);
    }
  }
T expanding(const T &new_value) {
    if(std::isnan(new_value)){
        nan_count ++;
    }else
    {
        sum = sum + new_value;
    }
    step++;
    data[index] = new_value;
    index = (index + 1) % window;
    if(step < window) return std::nan("");
    if(step == nan_count){
        return new_value;
    }else
    {
        return sum / (step - nan_count);
    }
}

T calculate(const T &new_value) {
  if (step < window) {
    return expanding(new_value);
  } else {
    return rolling(new_value);
  }
}
T getSum() const {
  return sum;
}
bool inited() const {
  return data != nullptr;
}
void dump() const {
  printf("MEAN_OBJ: sum=%f, window=%d, step=%d, index=%d, nan_count=%d", sum, window, step, index, nan_count);
}
private:
  T sum;
  int window;
  int step;
  int index;
  T *data = nullptr;
  int nan_count = 0;
};
)";
    os_ << "// global state\n"
        << "TS_MEAN_CY<float> g_ts_mean_state[1000];\n"
        << "/** see C_backend_api.h#TVMBackendPackedCFunc\n"
        << " * \\param TVMValue *args The arguments\n"
        << " * \\param int *type_codes The type codes of the arguments\n"
        << " * \\param int num_args Number of arguments\n"
        << " * \\param TVMValue *out_ret_value The output value of the return value\n"
        << " * \\param int *out_ret_tcode The output type code of the return value\n"
        << " * \\param void *resource_handle Pointer to associated resource\n"
        << " */\n"
        << " extern \"C\" int " << fnname << "(\n"
        << "        void *args,\n" 
        << "        int *type_codes,\n"
        << "        int num_args,\n"
        << "        void *out_ret_value,\n"
        << "        int *out_ret_tcode,\n"
        << "        void *resource_handle) {\n";
    os_ << R"(
    DLTensor *data = (DLTensor*)(((TVMValue*)args)[0].v_handle);
    float *inputdata = (float*)data->data;
    DLTensor *window = (DLTensor*)(((TVMValue*)args)[1].v_handle);
    assert(window->ndim == 1);
    int *windowdata = (int*)window->data;
    int window_size = windowdata[0];
    printf("tsmean: window: %d\n", window_size);
    assert(window_size > 0);
    DLTensor *output = (DLTensor*)(((TVMValue*)args)[2].v_handle);
    float *outdata = (float*)output->data;
    const int TS_DATA_LEN = data->shape[0];
    assert(TS_DATA_LEN > 0);
    if (!g_ts_mean_state[0].inited()) {
      for (int i = 0; i < TS_DATA_LEN; ++i) {
        g_ts_mean_state[i].init(window_size);
      }
    }
    )";
    os_ << "  printf(\"args=%p, type_codes=%p, num_args=%d, out_ret_value=%p, out_ret_tcode=%p, resource_handle=%p\\n\",\n"
        << "         args, type_codes, num_args, out_ret_value, out_ret_tcode, resource_handle); \n"
        << "  for (int n = 0; n < num_args; ++n) {\n"
        << "    printf(\"codegen: === type[%d]=%d, args[%d]=%p\\n\", n, type_codes[n], n, ((TVMValue*)args)[n]);\n"
        << "    DLTensor *data = (DLTensor*)(((TVMValue*)args)[n].v_handle);\n"
        << "    printf(\"codegen: data=%p, data->data=%p, data->ndim=%d\\n\", data, data->data, data->ndim);\n"
        << "    float *outdata = (float*)data->data;\n"
        << "    printf(\"codegen: outdata=%p, outdata[0]=%d, ndim=%d\\n\", outdata, outdata[0], data->ndim);\n"
        << "    for (int m = 0; m < data->ndim; ++m) {\n"
        << "      printf(\"codegen: - shape[%d]=%d\\n\", m, data->shape[m]);\n"
        << "    }\n"
        << "  }\n"
        << "  for (int i = 0; i < data->shape[0]; ++i) {\n"
        << "    outdata[i] = g_ts_mean_state[i].calculate(inputdata[i]);\n"
        << "  }\n"
        << "    printf(\"tsmean: \");\n"
        << "    g_ts_mean_state[0].dump();\n"
        << "    printf(\"\\n\");"
        << "  return 0;\n"
        << "}\n";
  }

  void GenerateCodeAxisAbs(const tvm::tir::PrimFunc& func, const std::string& fnname) {
    os_ << "// CodeGen for function: " << fnname << "\n";
    const auto *op = func.operator->();
    const auto &signature = op->func_type_annotation();
    os_ << "// Signature: " << signature << "\n";
    os_ << R"CPP(
#include <iostream>
#include <cmath>
#include <cassert>
#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
using namespace std;

template<typename T>
class TS_MEAN_CY {
public:
  TS_MEAN_CY() = default;
  void init(int window) {
    this->window = window;
    this->step = 0;
    this->index = 0;
    this->data = new T[window];
  }
  ~TS_MEAN_CY() { delete[] data; }
  T rolling(const T &new_value) {
    if (std::isnan(new_value)) {
      nan_count++;
    } else{       sum = sum + new_value;
    }
    if (std::isnan(data[index])) {
      nan_count--;
    } else {
      sum = sum - data[index];
    }
    data[index] = new_value;
    index = (index + 1) % window;
    if (window == nan_count) {
      return new_value;
    } else {
      return sum / (window - nan_count);
    }
  }
T expanding(const T &new_value) {
    if(std::isnan(new_value)){
        nan_count ++;
    }else
    {
        sum = sum + new_value;
    }
    data[index] = new_value;
    index = (index + 1) % window;
    if(step < window) return std::nan("");
    if(step == nan_count){
        return new_value;
    }else
    {
        return sum / (step - nan_count);
    }
}

T calculate(const T &new_value) {
  step++;
  if (step <= window) {
    return expanding(new_value);
  } else {
    return rolling(new_value);
  }
}
T getSum() const {
  return sum;
}
bool inited() const {
  return data != nullptr;
}
private:
  T sum;
  int window;
  int step;
  int index;
  T *data = nullptr;
  int nan_count = 0;
};
// global state
TS_MEAN_CY<float> g_ts_mean_state[1000];/** see C_backend_api.h#TVMBackendPackedCFunc
 * \param TVMValue *args The arguments
 * \param int *type_codes The type codes of the arguments
 * \param int num_args Number of arguments
 * \param TVMValue *out_ret_value The output value of the return value
 * \param int *out_ret_tcode The output type code of the return value
 * \param void *resource_handle Pointer to associated resource
 */
)CPP";
    os_ << "extern \"C\" int " << fnname << "(\n";
    os_ << R"CPP(
        void *args,
        int *type_codes,
        int num_args,
        void *out_ret_value,
        int *out_ret_tcode,
        void *resource_handle) {

    DLTensor *data = (DLTensor*)(((TVMValue*)args)[0].v_handle);
    float *inputdata = (float*)data->data;
    DLTensor *window = (DLTensor*)(((TVMValue*)args)[1].v_handle);
    assert(window->ndim == 1);
    int *windowdata = (int*)window->data;
    assert(windowdata[0] > 0);
    DLTensor *output = (DLTensor*)(((TVMValue*)args)[2].v_handle);
    float *outdata = (float*)output->data;
    const int TS_DATA_LEN = data->shape[0];
    assert(TS_DATA_LEN > 0);
    if (!g_ts_mean_state[0].inited()) {
      for (int i = 0; i < TS_DATA_LEN; ++i) {
        g_ts_mean_state[i].init(0);
      }
    }
      printf("args=%p, type_codes=%p, num_args=%d, out_ret_value=%p, out_ret_tcode=%p, resource_handle=%p\n",
         args, type_codes, num_args, out_ret_value, out_ret_tcode, resource_handle); 
  for (int n = 0; n < num_args; ++n) {
    printf("codegen: === type[%d]=%d, args[%d]=%p\n", n, type_codes[n], n, &((TVMValue*)args)[n]);
    DLTensor *data = (DLTensor*)(((TVMValue*)args)[n].v_handle);
    printf("codegen: data=%p, data->data=%p, data->ndim=%d\n", data, data->data, data->ndim);
    float *outdata = (float*)data->data;
    printf("codegen: outdata=%p, outdata[0]=%f, ndim=%d\n", outdata, outdata[0], data->ndim);
    for (int m = 0; m < data->ndim; ++m) {
      printf("codegen: - shape[%d]=%ld\n", m, data->shape[m]);
    }
  }
  for (int i = 0; i < data->shape[0]; ++i) {
    outdata[i] = g_ts_mean_state[i].calculate(inputdata[i]);
    printf("tsmean: %d: (%f)->%f\n", i, inputdata[i], outdata[i]);
  }
  return 0;
}
        )CPP";
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

runtime::Module MyCompiler(const ObjectRef& ref) {
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

TVM_REGISTER_GLOBAL("relay.ext.custom").set_body_typed(MyCompiler);
TVM_REGISTER_GLOBAL("target.build.custom")
    .set_body_typed([](IRModule mod, Target target) -> runtime::Module {
      return MyCompiler(mod);
    });
}  // namespace contrib
}  // namespace relay
}  // namespace tvm 