set(
    LLVM_TARGET_DEFINITIONS
        AtemIROps.td
)
add_mlir_dialect(AtemIR atemir)

mlir_tablegen(AtemIRAttrDefs.h.inc -gen-attrdef-decls
        -attrdefs-dialect=atemir)
mlir_tablegen(AtemIRAttrDefs.cpp.inc -gen-attrdef-defs
        -attrdefs-dialect=atemir)
add_public_tablegen_target(MLIRAtemIRAttrDefsIncGen)

mlir_tablegen(AtemIROpEnums.h.inc -gen-enum-decls
        --dialect=atemir)
mlir_tablegen(AtemIROpEnums.cpp.inc -gen-enum-defs
        --dialect=atemir)
add_public_tablegen_target(MLIRAtemIREnumsIncGen)