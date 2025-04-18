#include "tvm/runtime/module.h"
#include "tvm/runtime/registry.h"
namespace tvm {
runtime::Module DXTCompiler() {
    static_assert(true, "haha");
    CHECK(false) << "Not implemented yet";
    return {

    };
}

TVM_REGISTER_GLOBAL("relay.ext.dxtc").set_body_typed(DXTCompiler);
}