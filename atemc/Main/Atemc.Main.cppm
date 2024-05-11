module;

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

#include "antlr4-runtime.h"

#include "AtemParser.h"
#include "AtemLexer.h"

#include "AtemIR/Dialect/include/AtemIRDialect.h"

#include <string>
#include <memory>

export module Atemc.Main;

import Atemc.Exceptions;
import Atemc.Lexer;
import Atemc.Parser;

namespace atemc::main
{
    namespace cl = llvm::cl;

    static cl::opt<std::string> inputFilename(
        cl::Positional,
        cl::desc("<input Atem file>"),
        cl::init("-"),
        cl::value_desc("filename")
    );

    namespace
    {
        enum InputType { AtemSource, MLIR };
    }

    static cl::opt<enum InputType> inputType(
        "input",
        cl::init(AtemSource),
        cl::desc("Decided the kind of input file"),
        cl::values(clEnumValN(AtemSource, "atemsource", "load the input file as an Atem source.")),
        cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file."))
    );

    namespace
    {
        enum Action
        {
            None,
            DumpAST,
            DumpAtemIR,
            DumpMLIRStandard,
            DumpMLIRLLVM,
            DumpLLVMIR,
            RunJIT
        };
    }
    static cl::opt<enum Action> emitAction(
        "emit",
        cl::desc("Select the kind of output desired"),
        cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
        cl::values(clEnumValN(DumpAtemIR, "atemir", "output the Atem IR dump")),
        cl::values(clEnumValN(DumpMLIRStandard, "mlir-standard",
                              "output the MLIR dump after lowering to standard")),
        cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                              "output the MLIR dump after llvm lowering")),
        cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
        cl::values(clEnumValN(RunJIT, "jit", "JIT the code and run it by invoking the main function"))
    );

    static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
}

export namespace atemc::main
{
    auto parseAST() -> std::unique_ptr<mlir::ModuleOp>
    {
    }

    auto dumpAtemIR(mlir::MLIRContext &context) -> int
    {
        antlr4::ANTLRFileStream file_stream;
        file_stream.loadFromFile(inputFilename);
        atemc_antlr::AtemLexer atem_lexer(&file_stream);
        antlr4::CommonTokenStream tokens(&atem_lexer);
        atemc_antlr::AtemParser atem_parser(&tokens);

        atemc::lexer::antlr4::AtemLexerErrorListener lexer_error_listener;
        atemc::parser::antlr4::AtemParserErrorListener parser_error_listener;

        atem_lexer.removeErrorListeners();
        atem_parser.removeErrorListeners();
        atem_parser.removeParseListeners();

        atem_lexer.addErrorListener(&lexer_error_listener);
        atem_parser.addErrorListener(&parser_error_listener);


        atemc_antlr::AtemParser::ProgramContext* tree =
            atem_parser.program();

        parser::antlr4::AtemIRBuilderVisitor visitor(tree, context);
        auto ir = visitor.buildModule();

        ir->dump();

        return 0;
    }

    auto dumpAST() -> int
    {
        if (inputType == InputType::MLIR)
        {
            llvm::errs() << "Can't dump AST for MLIR input files\n";
            return -1;
        }

        antlr4::ANTLRFileStream file_stream;
        file_stream.loadFromFile(inputFilename);
        atemc_antlr::AtemLexer atem_lexer(&file_stream);
        antlr4::CommonTokenStream tokens(&atem_lexer);
        atemc_antlr::AtemParser atem_parser(&tokens);

        atemc::lexer::antlr4::AtemLexerErrorListener lexer_error_listener;
        atemc::parser::antlr4::AtemParserErrorListener parser_error_listener;

        atem_lexer.removeErrorListeners();
        atem_parser.removeErrorListeners();
        atem_parser.removeParseListeners();

        atem_lexer.addErrorListener(&lexer_error_listener);
        atem_parser.addErrorListener(&parser_error_listener);

        atemc_antlr::AtemParser::ProgramContext* tree =
            atem_parser.program();
        auto ast_str = tree->toStringTree(true);
        llvm::outs() << "Abstract Syntax Tree:\n" << ast_str;
        return 0;
    }

    auto atemcMain(int argc, const char* argv[]) -> int
    {
        mlir::registerAsmPrinterCLOptions();
        mlir::registerMLIRContextCLOptions();
        mlir::registerPassManagerCLOptions();

        cl::ParseCommandLineOptions(argc, argv, "Experimental Atem Compiler\n");

        if (emitAction == Action::DumpAST)
        {
            return dumpAST();
        }

        mlir::DialectRegistry registry;
        mlir::MLIRContext context(registry);
        context.getOrLoadDialect<atemir::AtemIRDialect>();

        if (emitAction == Action::DumpAtemIR)
        {
            return dumpAtemIR(context);
        }

        return 0;
    }
}

export auto main(int argc, const char* argv[]) -> int
{
    return atemc::main::atemcMain(argc, argv);
}