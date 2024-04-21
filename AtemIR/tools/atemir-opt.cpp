#define _ITERATOR_DEBUG_LEVEL 0

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
//#include "include/AtemDialect.h"

using namespace mlir;
using namespace llvm;

int main(int argc, char ** argv) {
    DialectRegistry registry;
    // 注册 Dialect
    registry.insert<func::FuncDialect>();
    // 注册两个 Pass
    registerCSEPass();
    registerCanonicalizerPass();
    return asMainReturnCode(MlirOptMain(argc, argv, "toy-opt", registry));
}