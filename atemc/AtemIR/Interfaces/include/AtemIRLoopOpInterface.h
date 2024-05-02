#ifndef ATEMIRLOOPOPINTERFACE_H
#define ATEMIRLOOPOPINTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace atemir {
    namespace detail {
        auto verifyLoopOpInterface(::mlir::Operation *Op) -> ::mlir::LogicalResult;
    }
}

#include "Interfaces/include/AtemIRLoopOpInterface.h.inc"

#endif //ATEMIRLOOPOPINTERFACE_H
