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
)

target_compile_features(atemc-exp PRIVATE cxx_std_23)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

target_sources(
        atemc-exp PRIVATE
        FILE_SET atemc_exp_cpp_modules TYPE CXX_MODULES FILES
        Exceptions/Atemc.Exceptions.cppm
        Main/Atemc.Main.cppm
        Lexer/Atemc.Lexer.cppm
        Lexer/ANTLR4Lexer/Atemc.Lexer.ANTLR4Lexer.cppm
        Lexer/ANTLR4Lexer/Atemc.Lexer.ANTLR4Lexer.ErrorListener.cppm
        Parser/Atemc.Parser.cppm
        Parser/ANTLR4Parser/Atemc.Parser.ANTLR4Parser.cppm
        Parser/ANTLR4Parser/Atemc.Parser.ANTLR4Parser.Listener.cppm
        Parser/ANTLR4Parser/Atemc.Parser.ANTLR4Parser.Visitor.cppm
        Parser/ANTLR4Parser/Atemc.Parser.ANTLR4Parser.ErrorListener.cppm
        Utils/Atemc.Utils.cppm
        Utils/Atemc.Utils.Unimplemented.cppm
)

target_link_libraries(
        atemc-exp PRIVATE
        antlr4_shared

        AtemIR
        AtemIRInterfaces

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

include_directories(${CMAKE_CURRENT_BINARY_DIR}/AtemIR)

add_subdirectory(AtemIR)

find_package(Backward CONFIG REQUIRED)
target_link_libraries(atemc-exp PRIVATE Backward::Backward)

find_package(libdwarf CONFIG REQUIRED)
target_link_libraries(atemc-exp PRIVATE libdwarf::dwarf)