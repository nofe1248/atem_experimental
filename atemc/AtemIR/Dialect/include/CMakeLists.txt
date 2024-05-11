set(
    LLVM_TARGET_DEFINITIONS
        AtemIROps.td
)
add_mlir_dialect(AtemIR atemir)
add_mlir_doc(AtemIROps AtemIROps Dialects/ -gen-op-doc)
add_mlir_doc(AtemIRAttrs AtemIRAttrs Dialects/ -gen-attrdef-doc)
add_mlir_doc(AtemIRTypes AtemIRTypes Dialects/ -gen-typedef-doc)

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