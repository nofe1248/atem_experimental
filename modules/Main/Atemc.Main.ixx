module;

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

export module Atemc.Main;

using namespace mlir;

export auto main(int argc, const char* argv[]) -> int
{
    MLIRContext ctx;
    ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();
    auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);
    src->print(llvm::outs());
    src->dump();
    return 0;
}