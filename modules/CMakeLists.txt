add_executable(atemc-exp)

set_property(TARGET atemc-exp PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

target_sources(
        atemc-exp PRIVATE
        FILE_SET atemc_exp_cpp_modules TYPE CXX_MODULES FILES
        Main/Atemc.Main.ixx
)

target_link_libraries(
        atemc-exp PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIRAnalysis
        MLIRBuiltinToLLVMIRTranslation
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRFunctionInterfaces
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms
)