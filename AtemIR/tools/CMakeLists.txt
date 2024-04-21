add_mlir_tool(atemir-opt atemir-opt.cpp)

target_link_libraries(
        atemir-opt PRIVATE
        MLIRIR MLIRParser MLIRSupport
        AtemIR               # 对应 #include "toy/ToyDialect.h"
        MLIROptLib        # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
        MLIRFuncDialect   # 对应 #include "mlir/Dialect/Func/IR/FuncOps.h"
        MLIRTransforms    # 对应 #include "mlir/Transforms/Passes.h"
)