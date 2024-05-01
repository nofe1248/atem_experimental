#ifndef ATEMIRATTRS_H
#define ATEMIRATTRS_H

#include "AtemIR/Dialect/include/AtemIRTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/APInt.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/include/AtemIRAttrDefs.h.inc"

#endif //ATEMIRATTRS_H
