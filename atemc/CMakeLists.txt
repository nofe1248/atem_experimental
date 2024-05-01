include(${antlr_SOURCE_DIR}/runtime/Cpp/cmake/FindANTLR.cmake)

antlr_target(
        AtemLexer ${PROJECT_SOURCE_DIR}/atemc/Grammar/AtemLexer.g4 LEXER
        PACKAGE atemc_antlr
)
antlr_target(
        AtemParser ${PROJECT_SOURCE_DIR}/atemc/Grammar/AtemParser.g4 PARSER
        PACKAGE atemc_antlr
        DEPENDS_ANTLR AtemLexer
        COMPILE_FLAGS -lib ${ANTLR_AtemLexer_OUTPUT_DIR}
        VISITOR
        LISTENER
)

include_directories(${ANTLR_AtemLexer_OUTPUT_DIR})
include_directories(${ANTLR_AtemParser_OUTPUT_DIR})
include_directories(${antlr_SOURCE_DIR}/runtime/Cpp/runtime/src)

add_executable(
        atemc-exp
        ${ANTLR_AtemLexer_CXX_OUTPUTS}
        ${ANTLR_AtemParser_CXX_OUTPUTS}
        AtemIR/Dialect/lib/AtemIR.cpp
        AtemIR/Dialect/lib/AtemIROps.cpp
        AtemIR/Dialect/lib/AtemIRAttrs.cpp
        AtemIR/Dialect/lib/AtemIRTypes.cpp
)

target_compile_features(atemc-exp PRIVATE cxx_std_23)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

target_sources(
        atemc-exp PRIVATE
        FILE_SET atemc_exp_cpp_modules TYPE CXX_MODULES FILES
        Main/Atemc.Main.cppm
        Lexer/Atemc.Lexer.cppm
        Lexer/ANTLR4Lexer/Atemc.Lexer.ANTLR4Lexer.cppm
        Parser/Atemc.Parser.cppm
        Parser/ANTLR4Parser/Atemc.Parser.ANTLR4Parser.cppm
)

target_link_libraries(
        atemc-exp PRIVATE
        antlr4_shared

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

include_directories(${CMAKE_CURRENT_BINARY_DIR}/AtemIR/include)

add_subdirectory(AtemIR)