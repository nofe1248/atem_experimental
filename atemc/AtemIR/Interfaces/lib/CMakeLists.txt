add_mlir_library(
        AtemIRInterfaces
        AtemIRFPTypeInterface.cpp
        AtemIRLoopOpInterface.cpp

        DEPENDS
        MLIRAtemIRFPTypeInterfaceIncGen
        MLIRAtemIRLoopOpInterfaceIncGen
)