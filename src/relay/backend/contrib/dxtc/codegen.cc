#include "tvm/runtime/module.h"
#include "tvm/runtime/registry.h"
#include <iostream>
namespace tvm {
runtime::Module DXTCompiler() {
    std::cout << "Not implemented yet" << std::endl;
    return {
        
    };
}

TVM_REGISTER_GLOBAL("relay.ext.dxtc").set_body_typed(DXTCompiler);
}