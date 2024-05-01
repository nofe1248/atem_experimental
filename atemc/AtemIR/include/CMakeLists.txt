set(
    LLVM_TARGET_DEFINITIONS 
    AtemIROps.td
)
add_mlir_dialect(AtemIR AtemIR)
mlir_tablegen(AtemIRAttrDefs.h.inc -gen-attrdef-decls
        -attrdefs-dialect=AtemIR)
mlir_tablegen(AtemIRAttrDefs.cpp.inc -gen-attrdef-defs
        -attrdefs-dialect=AtemIR)