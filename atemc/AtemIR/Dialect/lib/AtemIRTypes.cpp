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
#include "AtemIR/Dialect/include/AtemIRTypes.h"

using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "Dialect/include/AtemIRTypes.cpp.inc"

using namespace mlir;
using namespace atemir;

auto AtemIRDialect::registerTypes() -> void
{
    // Register tablegen'd types.
    addTypes<
        #define GET_TYPEDEF_LIST
        #include "Dialect/include/AtemIRTypes.cpp.inc"
    >();
}

auto AtemIRDialect::parseType(DialectAsmParser &parser) const -> Type
{
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    StringRef mnemonic;
    Type genType;

    // Try to parse as a tablegen'd type.
    OptionalParseResult parseResult =
        generatedTypeParser(parser, &mnemonic, genType);
    if (parseResult.has_value())
        return genType;

    return StringSwitch<function_ref<Type()>>(mnemonic)
        .Default([&] {
            parser.emitError(typeLoc) << "unknown Atem IR type: " << mnemonic;
            return Type();
        })();
}

auto AtemIRDialect::printType(Type type, DialectAsmPrinter &os) const -> void
{
    // Try to print as a tablegen'd type.
    if (generatedTypePrinter(type, os).succeeded())
        return;

    TypeSwitch<Type>(type)
        .Default([](Type) {
            llvm::report_fatal_error("printer is missing a handler for this type");
        });
}

auto atemir::IntegerType::parse(AsmParser &parser) -> Type
{
    auto *context = parser.getBuilder().getContext();
    auto loc = parser.getCurrentLocation();
    bool isSigned = true;
    unsigned width = 64;

    if (parser.parseLess())
    {
        return {};
    }

    llvm::StringRef sign;
    if (parser.parseKeyword(&sign))
    {
        return {};
    }
    if (sign.equals("s"))
    {
        isSigned = true;
    }
    else if (sign.equals("u"))
    {
        isSigned = false;
    }
    else
    {
        parser.emitError(loc, "expected 's' or 'u'");
        return {};
    }

    if (parser.parseComma())
    {
        return {};
    }

    if (parser.parseInteger(width))
    {
        return {};
    }
    if (width < 1)
    {
        parser.emitError(loc, "expected integer width to be no less than 1");
        return {};
    }

    if(parser.parseGreater())
    {
        return {};
    }

    return IntegerType::get(context, width, isSigned);
}

auto atemir::FunctionType::clone(::mlir::TypeRange inputs, ::mlir::TypeRange results) const -> ::atemir::FunctionType
{
    return atemir::FunctionType::get(llvm::to_vector(inputs), llvm::to_vector(results));
}

auto atemir::FunctionType::isReturningUnit() const -> bool
{
    if (getResults().size() == 1)
    {
        if(mlir::isa<atemir::UnitType>(getResults().front()))
        {
            return true;
        }
    }
    return false;
}

auto atemir::IntegerType::print(mlir::AsmPrinter &printer) const -> void
{
    auto sign = this->getIsSigned() ? 's' : 'u';
    printer << '<' << sign << ", " << this->getWidth() << '>';
}

auto atemir::IntegerType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const -> llvm::TypeSize {
    return llvm::TypeSize::getFixed(getWidth());
}

auto atemir::IntegerType::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                          ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return (uint64_t)(getWidth() / 8);
}

auto atemir::IntegerType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return (uint64_t)(getWidth() / 8);
}

auto atemir::BooleanType::parse(AsmParser &parser) -> Type
{
    return get(parser.getContext());
}

auto atemir::BooleanType::print(mlir::AsmPrinter &printer) const -> void
{

}

auto atemir::BooleanType::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const -> llvm::TypeSize {
    return llvm::TypeSize::getFixed(1);
}

auto atemir::BooleanType::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                          ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return 1;
}

auto atemir::BooleanType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return 1;
}

auto atemir::Float16Type::getFloatSemantics() const -> const llvm::fltSemantics &
{
    return llvm::APFloat::IEEEhalf();
}

auto atemir::Float16Type::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const -> llvm::TypeSize {
    return llvm::TypeSize::getFixed(getWidth());
}

auto atemir::Float16Type::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                          ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return (uint64_t)(getWidth() / 8);
}

auto atemir::Float16Type::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return (uint64_t)(getWidth() / 8);
}

auto atemir::Float32Type::getFloatSemantics() const -> const llvm::fltSemantics &
{
    return llvm::APFloat::IEEEsingle();
}

auto atemir::Float32Type::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const -> llvm::TypeSize {
    return llvm::TypeSize::getFixed(getWidth());
}

auto atemir::Float32Type::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                          ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return (uint64_t)(getWidth() / 8);
}

auto atemir::Float32Type::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return (uint64_t)(getWidth() / 8);
}

auto atemir::Float64Type::getFloatSemantics() const -> const llvm::fltSemantics &
{
    return llvm::APFloat::IEEEdouble();
}

auto atemir::Float64Type::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const -> llvm::TypeSize {
    return llvm::TypeSize::getFixed(getWidth());
}

auto atemir::Float64Type::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                          ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return (uint64_t)(getWidth() / 8);
}

auto atemir::Float64Type::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return (uint64_t)(getWidth() / 8);
}

auto atemir::Float80Type::getFloatSemantics() const -> const llvm::fltSemantics &
{
    return llvm::APFloat::x87DoubleExtended();
}

auto atemir::Float80Type::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const -> llvm::TypeSize {
    return llvm::TypeSize::getFixed(16);
}

auto atemir::Float80Type::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                          ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return 16;
}

auto atemir::Float80Type::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return 16;
}

auto atemir::Float128Type::getFloatSemantics() const -> const llvm::fltSemantics &
{
    return llvm::APFloat::IEEEquad();
}

auto atemir::Float128Type::getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                              mlir::DataLayoutEntryListRef params) const -> llvm::TypeSize {
    return llvm::TypeSize::getFixed(getWidth());
}

auto atemir::Float128Type::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                          ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return (uint64_t)(getWidth() / 8);
}

auto atemir::Float128Type::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                ::mlir::DataLayoutEntryListRef params) const -> uint64_t {
    return (uint64_t)(getWidth() / 8);
}

auto atemir::PointerType::getTypeSizeInBits(const ::mlir::DataLayout &dataLayout,
                               ::mlir::DataLayoutEntryListRef params) const -> llvm::TypeSize
{
    return llvm::TypeSize::getFixed(64);
}

auto atemir::PointerType::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                                          ::mlir::DataLayoutEntryListRef params) const -> uint64_t
{
    return 8;
}

auto atemir::PointerType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                          ::mlir::DataLayoutEntryListRef params) const -> uint64_t
{
    return 8;
}