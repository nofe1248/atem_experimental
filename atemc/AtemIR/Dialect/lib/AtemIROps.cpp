#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Location.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/STLExtras.h"

#include "AtemIR/Dialect/include/AtemIRDialect.h"
#include "AtemIR/Dialect/include/AtemIROps.h"

using namespace mlir;


static ParseResult parseConstantValue(OpAsmParser &parser,
mlir::Attribute &valueAttr);
static void printConstantValue(OpAsmPrinter &p, atemir::ConstantOp op,
                               Attribute value);

#define GET_OP_CLASSES
#include "Dialect/include/AtemIR.cpp.inc"

using namespace mlir;
using namespace atemir;

void AtemIRDialect::registerOperations() {
    // Register tablegen'd operations.
    addOperations<
        #define GET_OP_LIST
        #include "Dialect/include/AtemIR.cpp.inc"
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

mlir::ParseResult FunctionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    auto buildFuncType = [](auto & builder, auto argTypes, auto results, auto, auto) {
        return builder.getFunctionType(argTypes, results);
    };
    return function_interface_impl::parseFunctionOp(
      parser, result, false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name)
    );
}

void FunctionOp::print(mlir::OpAsmPrinter &p) {
    // Dispatch to the FunctionOpInterface provided utility method that prints the
    // function operation.
    mlir::function_interface_impl::printFunctionOp(
        p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
}

mlir::LogicalResult ConstantOp::inferReturnTypes(
  mlir::MLIRContext * context,
  std::optional<mlir::Location> location,
  Adaptor adaptor,
  llvm::SmallVectorImpl<mlir::Type> & inferedReturnType
) {
    auto type = adaptor.getValueAttr().getType();
    inferedReturnType.push_back(type);
    return mlir::success();
}