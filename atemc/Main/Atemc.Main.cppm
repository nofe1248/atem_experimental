module;

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "llvm/Support/raw_ostream.h"

#include "AtemIR/Dialect/include/AtemIRDialect.h"

export module Atemc.Main;

using namespace mlir;

export auto main(int argc, const char* argv[]) -> int
{
    MLIRContext ctx;
    ctx.loadDialect<func::FuncDialect, arith::ArithDialect, atemir::AtemIRDialect, scf::SCFDialect, async::AsyncDialect>();

    auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);
    src->print(llvm::outs());
    src->dump();

    return 0;
}