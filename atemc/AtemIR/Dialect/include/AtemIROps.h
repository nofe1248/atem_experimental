#ifndef ATEMIROPS_H
#define ATEMIROPS_H

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/IR/RegionKindInterface.h"

#include "AtemIR/Dialect/include/AtemIRDialect.h"
#include "AtemIR/Dialect/include/AtemIRTypes.h"

#include "AtemIR/Interfaces/include/AtemIRLoopOpInterface.h"

namespace atemir {
    auto buildTerminatedBody(mlir::OpBuilder &builder, mlir::Location loc) -> void;
}

#define GET_OP_CLASSES
#include "Dialect/include/AtemIR.h.inc"

#endif //ATEMIROPS_H
