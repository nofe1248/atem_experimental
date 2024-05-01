#ifndef ATEMIRTYPES_H
#define ATEMIRTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "AtemIR/Interfaces/AtemIRFPTypeInterface.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/include/AtemIRTypes.h.inc"

namespace atemir
{
namespace detail
{

}
}

#endif //ATEMIRTYPES_H
