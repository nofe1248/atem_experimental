#include "AtemIR/include/AtemIRDialect.h"
#include "AtemIR/include/AtemIROps.h"
#include "AtemIR/include/AtemIRTypes.h"
#include "AtemIRDialect.cpp.inc"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

static ParseResult parseConstantValue(OpAsmParser &parser,
mlir::Attribute &valueAttr);
static void printConstantValue(OpAsmPrinter &p, atemir::ConstantOp op,
                               Attribute value);

#define GET_OP_CLASSES
#include "AtemIR.cpp.inc"
using namespace atemir;
void AtemIRDialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "AtemIR.cpp.inc"
    >();
}

static ParseResult parseConstantValue(OpAsmParser &parser,
mlir::Attribute &valueAttr) {
    NamedAttrList attr;
    return parser.parseAttribute(valueAttr, "value", attr);
}

static void printConstant(OpAsmPrinter &p, Attribute value) {
    p.printAttribute(value);
}

static void printConstantValue(OpAsmPrinter &p, atemir::ConstantOp op,
                               Attribute value) {
    printConstant(p, value);
}