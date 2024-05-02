add_mlir_library(
        AtemIR
        AtemIR.cpp
        AtemIRTypes.cpp
        AtemIRAttrs.cpp
        AtemIROps.cpp

        DEPENDS
        MLIRAtemIRIncGen
        AtemIRAttrDefsIncGen
)