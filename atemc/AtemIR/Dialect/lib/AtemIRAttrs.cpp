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

using namespace mlir;
using namespace llvm;

static auto printFloatLiteral(mlir::AsmPrinter &p, llvm::APFloat value, mlir::Type ty) -> void;
static auto parseFloatLiteral(mlir::AsmParser &parser, mlir::FailureOr<llvm::APFloat> &value, mlir::Type ty)
    -> mlir::ParseResult;

#include "AtemIR/Dialect/include/AtemIRDialect.h"
#include "AtemIR/Dialect/include/AtemIRAttrs.h"
#include "AtemIR/Dialect/include/AtemIRTypes.h"

#include "AtemIR/Interfaces/include/AtemIRFPTypeInterface.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/include/AtemIRAttrDefs.cpp.inc"

using namespace atemir;

auto AtemIRDialect::registerAttributes() -> void
{
    addAttributes<
        #define GET_ATTRDEF_LIST
        #include "Dialect/include/AtemIRAttrDefs.cpp.inc"
    >();
}

auto AtemIRDialect::parseAttribute(DialectAsmParser &parser, Type type) const -> Attribute
{
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    StringRef mnemonic;
    Attribute genAttr;
    OptionalParseResult parseResult =
        generatedAttributeParser(parser, &mnemonic, type, genAttr);
    if (parseResult.has_value())
    {
        return genAttr;
    }
    parser.emitError(typeLoc, "unknown attribute in CIR dialect");
    return {};
}

auto AtemIRDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const -> void
{
    if (failed(generatedAttributePrinter(attr, os)))
    {
        llvm_unreachable("unexpected CIR type kind");
    }
}

auto ::atemir::IntegerAttr::parse(::mlir::AsmParser &parser,
    ::mlir::Type odsType) -> ::mlir::Attribute
{
    mlir::APInt ap_int;
    auto loc = parser.getCurrentLocation();

    if(not mlir::isa<IntegerType>(odsType))
    {
        return {};
    }
    auto type = mlir::cast<IntegerType>(odsType);

    if(parser.parseLess())
    {
        return {};
    }

    if (parser.parseInteger(ap_int))
    {
        parser.emitError(loc, "expected integer value");
    }
    if(ap_int.getBitWidth() > type.getWidth())
    {
        parser.emitError(loc, "integer value too large for the given type");
    }

    if(parser.parseGreater())
    {
        return {};
    }

    return IntegerAttr::get(type, ap_int);
}

auto ::atemir::IntegerAttr::print(::mlir::AsmPrinter &printer) const -> void
{
    printer << '<';
    printer << getValue();
    printer << '>';
}

static auto printFloatLiteral(mlir::AsmPrinter &p, llvm::APFloat value, mlir::Type ty) -> void {
    p << value;
}

static auto parseFloatLiteral(mlir::AsmParser &parser, mlir::FailureOr<llvm::APFloat> &value, mlir::Type ty)
    -> mlir::ParseResult
{
    double rawValue;
    if (parser.parseFloat(rawValue)) {
        return parser.emitError(parser.getCurrentLocation(),
                                "expected floating-point value");
    }

    auto losesInfo = false;
    value.emplace(rawValue);

    auto tyFpInterface = mlir::dyn_cast<atemir::AtemIRFPTypeInterface>(ty);
    if (!tyFpInterface) {
        // Parsing of the current floating-point literal has succeeded, but the
        // given attribute type is invalid. This error will be reported later when
        // the attribute is being verified.
        return success();
    }

    value->convert(tyFpInterface.getFloatSemantics(),
                   llvm::RoundingMode::TowardZero, &losesInfo);
    return success();
}

