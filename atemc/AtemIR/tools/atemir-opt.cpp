#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
#include "AtemIR/include/AtemIRDialect.h"

using namespace mlir;
using namespace llvm;

auto main(int argc, char ** argv) -> int {
    DialectRegistry registry;
    registry.insert<func::FuncDialect, atemir::AtemIRDialect>();
    registerCSEPass();
    registerCanonicalizerPass();
    return asMainReturnCode(MlirOptMain(argc, argv, "atemir-opt", registry));
}