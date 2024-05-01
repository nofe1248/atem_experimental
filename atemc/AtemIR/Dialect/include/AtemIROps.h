#ifndef ATEMIROPS_H
#define ATEMIROPS_H

#pragma once

#include "AtemIRDialect.h"
#include "AtemIRTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/RegionKindInterface.h"

#define GET_OP_CLASSES
#include "Dialect/include/AtemIR.h.inc"

#endif //ATEMIROPS_H
