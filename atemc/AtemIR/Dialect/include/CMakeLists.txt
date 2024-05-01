set(
    LLVM_TARGET_DEFINITIONS
        AtemIROps.td
)
add_mlir_dialect(AtemIR atemir)
mlir_tablegen(AtemIRAttrDefs.h.inc -gen-attrdef-decls
        -attrdefs-dialect=atemir)
mlir_tablegen(AtemIRAttrDefs.cpp.inc -gen-attrdef-defs
        -attrdefs-dialect=atemir)
add_public_tablegen_target(AtemIRAttrDefsIncGen)