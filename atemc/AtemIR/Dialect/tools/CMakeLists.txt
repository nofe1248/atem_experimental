add_mlir_tool(atemir-opt atemir-opt.cpp)

target_link_libraries(
        atemir-opt PRIVATE
        MLIRIR MLIRParser MLIRSupport
        AtemIR
        AtemIRInterfaces
        MLIROptLib
        MLIRFuncDialect
        MLIRSCFDialect
        MLIRAsyncDialect
        MLIRArithDialect
        MLIRTransforms
)