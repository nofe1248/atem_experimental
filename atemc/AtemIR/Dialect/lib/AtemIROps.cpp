#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "AtemIR/Dialect/include/AtemIRDialect.h"
#include "AtemIR/Dialect/include/AtemIROps.h"

static auto parseConstantValue(mlir::OpAsmParser &parser, mlir::Attribute &valueAttr) -> mlir::ParseResult;
static auto printConstantValue(mlir::OpAsmPrinter &p, atemir::ConstantOp op, mlir::Attribute value) -> void;
static auto parseGlobalOpTypeAndInitialValue(mlir::OpAsmParser &parser, mlir::TypeAttr &typeAttr,
                                             mlir::Attribute &initialValueAttr, mlir::Region &initRegion,
                                             mlir::Region &deinitRegion) -> mlir::ParseResult;
static auto printGlobalOpTypeAndInitialValue(mlir::OpAsmPrinter &p, atemir::GlobalOp op, mlir::TypeAttr type,
                                             mlir::Attribute initAttr, mlir::Region &initRegion,
                                             mlir::Region &deinitRegion) -> void;
auto ensureRegionTerm(mlir::OpAsmParser &parser, mlir::Region &region, mlir::SMLoc errLoc) -> mlir::LogicalResult;

#define GET_OP_CLASSES
#include "Dialect/include/AtemIR.cpp.inc"

auto atemir::AtemIRDialect::registerOperations() -> void
{
    // Register tablegen'd operations.
    addOperations<
#define GET_OP_LIST
#include "Dialect/include/AtemIR.cpp.inc"

        >();
}

namespace
{
template <typename Ty> struct EnumTraits
{
};
} // namespace

#define REGISTER_ENUM_TYPE(Ty)                                                                                         \
    template <> struct EnumTraits<Ty>                                                                                  \
    {                                                                                                                  \
        static mlir::StringRef stringify(Ty value)                                                                     \
        {                                                                                                              \
            return stringify##Ty(value);                                                                               \
        }                                                                                                              \
        static unsigned getMaxEnumVal()                                                                                \
        {                                                                                                              \
            return getMaxEnumValFor##Ty();                                                                             \
        }                                                                                                              \
    }
#define REGISTER_ENUM_TYPE_WITH_NS(NS, Ty)                                                                             \
    template <> struct EnumTraits<NS::Ty>                                                                              \
    {                                                                                                                  \
        static mlir::StringRef stringify(NS::Ty value)                                                                 \
        {                                                                                                              \
            return NS::stringify##Ty(value);                                                                           \
        }                                                                                                              \
        static unsigned getMaxEnumVal()                                                                                \
        {                                                                                                              \
            return NS::getMaxEnumValFor##Ty();                                                                         \
        }                                                                                                              \
    }

namespace
{
using namespace atemir;
REGISTER_ENUM_TYPE(GlobalLinkageKind);
} // namespace

static auto parseOptionalKeywordAlternative(mlir::AsmParser &parser, mlir::ArrayRef<mlir::StringRef> keywords) -> int
{
    for (auto en : llvm::enumerate(keywords))
    {
        if (mlir::succeeded(parser.parseOptionalKeyword(en.value())))
            return en.index();
    }
    return -1;
}

template <typename EnumTy, typename RetTy = EnumTy>
static auto parseOptionalAtemIRKeyword(mlir::AsmParser &parser, EnumTy defaultValue) -> RetTy
{
    mlir::SmallVector<mlir::StringRef, 10> names;
    for (unsigned i = 0, e = EnumTraits<EnumTy>::getMaxEnumVal(); i <= e; ++i)
        names.push_back(EnumTraits<EnumTy>::stringify(static_cast<EnumTy>(i)));

    int index = parseOptionalKeywordAlternative(parser, names);
    if (index == -1)
        return static_cast<RetTy>(defaultValue);
    return static_cast<RetTy>(index);
}

static auto parseConstantValue(mlir::OpAsmParser &parser, mlir::Attribute &valueAttr) -> mlir::ParseResult
{
    mlir::NamedAttrList attr;
    return parser.parseAttribute(valueAttr, "value", attr);
}

static auto printConstant(mlir::OpAsmPrinter &p, mlir::Attribute value) -> void
{
    p.printAttribute(value);
}

static auto printConstantValue(mlir::OpAsmPrinter &p, atemir::ConstantOp op, mlir::Attribute value) -> void
{
    printConstant(p, value);
}

static auto parseGlobalOpTypeAndInitialValue(mlir::OpAsmParser &parser, mlir::TypeAttr &typeAttr,
                                             mlir::Attribute &initialValueAttr, mlir::Region &initRegion,
                                             mlir::Region &deinitRegion) -> mlir::ParseResult
{
    mlir::Type opTy;
    if (parser.parseOptionalEqual().failed())
    {
        // Absence of equal means a declaration, so we need to parse the type.
        //  cir.global @a : i32
        if (parser.parseColonType(opTy))
            return mlir::failure();
    }
    else
    {
        // Parse contructor, example:
        //  cir.global @rgb = ctor : type { ... }
        if (!parser.parseOptionalKeyword("ctor"))
        {
            if (parser.parseColonType(opTy))
                return mlir::failure();
            auto parseLoc = parser.getCurrentLocation();
            if (parser.parseRegion(initRegion, /*arguments=*/{}, /*argTypes=*/{}))
                return mlir::failure();
            if (!initRegion.hasOneBlock())
                return parser.emitError(parser.getCurrentLocation(), "ctor region must have exactly one block");
            if (initRegion.back().empty())
                return parser.emitError(parser.getCurrentLocation(), "ctor region shall not be empty");
            if (ensureRegionTerm(parser, initRegion, parseLoc).failed())
                return mlir::failure();
        }
        else
        {
            // Parse constant with initializer, examples:
            //  cir.global @y = 3.400000e+00 : f32
            //  cir.global @rgb = #cir.const_array<[...] : !cir.array<i8 x 3>>
            if (parseConstantValue(parser, initialValueAttr).failed())
                return mlir::failure();

            assert(isa<mlir::TypedAttr>(initialValueAttr) && "Non-typed attrs shouldn't appear here.");
            auto typedAttr = cast<mlir::TypedAttr>(initialValueAttr);
            opTy = typedAttr.getType();
        }

        // Parse destructor, example:
        //   dtor { ... }
        if (!parser.parseOptionalKeyword("dtor"))
        {
            auto parseLoc = parser.getCurrentLocation();
            if (parser.parseRegion(deinitRegion, /*arguments=*/{}, /*argTypes=*/{}))
                return mlir::failure();
            if (!deinitRegion.hasOneBlock())
                return parser.emitError(parser.getCurrentLocation(), "dtor region must have exactly one block");
            if (deinitRegion.back().empty())
                return parser.emitError(parser.getCurrentLocation(), "dtor region shall not be empty");
            if (ensureRegionTerm(parser, deinitRegion, parseLoc).failed())
                return mlir::failure();
        }
    }

    typeAttr = mlir::TypeAttr::get(opTy);
    return mlir::success();
}

static auto printGlobalOpTypeAndInitialValue(mlir::OpAsmPrinter &p, atemir::GlobalOp op, mlir::TypeAttr type,
                                             mlir::Attribute initAttr, mlir::Region &initRegion,
                                             mlir::Region &deinitRegion) -> void
{
    auto printType = [&]() { p << ": " << type; };
    if (!op.isDeclaration())
    {
        p << "= ";
        if (!initRegion.empty())
        {
            p << "ctor ";
            printType();
            p << " ";
            p.printRegion(initRegion,
                          /*printEntryBlockArgs=*/false,
                          /*printBlockTerminators=*/false);
        }
        else
        {
            // This also prints the type...
            if (initAttr)
                printConstant(p, initAttr);
        }

        if (!deinitRegion.empty())
        {
            p << " dtor ";
            p.printRegion(deinitRegion,
                          /*printEntryBlockArgs=*/false,
                          /*printBlockTerminators=*/false);
        }
    }
    else
    {
        printType();
    }
}

static auto getLinkageAttrNameString() -> mlir::StringRef
{
    return "linkage";
}

auto atemir::FunctionOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::StringRef name,
                               FunctionType type, GlobalLinkageKind linkage, mlir::ArrayRef<mlir::NamedAttribute> attrs,
                               mlir::ArrayRef<mlir::DictionaryAttr> argAttrs) -> void
{
    state.addRegion();
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    state.addAttribute(getFunctionTypeAttrName(state.name), mlir::TypeAttr::get(type));
    state.addAttribute(getLinkageAttrNameString(), GlobalLinkageKindAttr::get(builder.getContext(), linkage));
    state.attributes.append(attrs.begin(), attrs.end());
    if (argAttrs.empty())
    {
        return;
    }

    mlir::function_interface_impl::addArgAndResultAttrs(
        builder, state, argAttrs, std::nullopt, getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

auto atemir::FunctionOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &state) -> mlir::ParseResult
{
    mlir::SMLoc loc = parser.getCurrentLocation();

    auto vis_name_attr = getSymVisibilityAttrName(state.name);

    state.addAttribute(getLinkageAttrNameString(),
                       GlobalLinkageKindAttr::get(
                           parser.getContext(),
                           parseOptionalAtemIRKeyword<GlobalLinkageKind>(parser, GlobalLinkageKind::ExternalLinkage)));
    mlir::StringRef vis_attr_str;
    if (parser.parseOptionalKeyword(&vis_attr_str, {"private", "public", "nested"}).succeeded())
    {
        state.addAttribute(vis_attr_str, parser.getBuilder().getStringAttr(vis_attr_str));
    }

    mlir::StringAttr name_attr;
    mlir::SmallVector<mlir::OpAsmParser::Argument, 8> arguments;
    mlir::SmallVector<mlir::DictionaryAttr, 1> result_attrs;
    mlir::SmallVector<mlir::Type, 8> arg_types;
    mlir::SmallVector<mlir::Type, 8> result_types;
    auto &builder = parser.getBuilder();

    if (parser.parseSymbolName(name_attr, mlir::SymbolTable::getSymbolAttrName(), state.attributes))
    {
        return mlir::failure();
    }

    bool is_variadic = false;
    if (mlir::function_interface_impl::parseFunctionSignature(parser, true, arguments, is_variadic, result_types,
                                                              result_attrs))
    {
        return mlir::failure();
    }

    for (auto &arg : arguments)
    {
        arg_types.push_back(arg.type);
    }

    if (result_types.empty())
    {
        result_types.push_back(atemir::UnitType::get(builder.getContext()));
    }

    auto func_type = atemir::FunctionType::get(arg_types, result_types);

    if (not func_type)
    {
        return mlir::failure();
    }

    state.addAttribute(getFunctionTypeAttrName(state.name), mlir::TypeAttr::get(func_type));

    if (parser.parseOptionalAttrDictWithKeyword(state.attributes))
    {
        return mlir::failure();
    }

    assert(result_attrs.size() == result_types.size());
    mlir::function_interface_impl::addArgAndResultAttrs(
        builder, state, arguments, result_attrs, getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));

    auto *body = state.addRegion();
    mlir::OptionalParseResult parse_result = parser.parseOptionalRegion(*body, arguments, false);
    if (parse_result.has_value())
    {
        if (failed(*parse_result))
        {
            return mlir::failure();
        }
        if (body->empty())
        {
            return parser.emitError(loc, "expected non-empty function body");
        }
    }
    return mlir::success();
}

auto atemir::FunctionOp::print(mlir::OpAsmPrinter &p) -> void
{
    p << " ";
    if (getLinkage() != GlobalLinkageKind::ExternalLinkage)
    {
        p << stringifyGlobalLinkageKind(getLinkage()) << ' ';
    }

    auto vis = getVisibility();
    if (vis != mlir::SymbolTable::Visibility::Public)
    {
        p << vis << " ";
    }

    p.printSymbolName(getSymName());
    auto func_type = getFunctionType();
    mlir::SmallVector<mlir::Type, 8> result_types;
    if (not func_type.isReturningUnit())
    {
        mlir::function_interface_impl::printFunctionSignature(p, *this, func_type.getInputs(), false,
                                                              func_type.getResults());
    }
    else
    {
        mlir::function_interface_impl::printFunctionSignature(p, *this, func_type.getInputs(), false, {});
    }
    mlir::function_interface_impl::printFunctionAttributes(p, *this,
                                                           {
                                                               getFunctionTypeAttrName(),
                                                               getLinkageAttrName(),
                                                               getSymVisibilityAttrName(),
                                                           });

    auto &body = getOperation()->getRegion(0);
    if (not body.empty())
    {
        p << ' ';
        p.printRegion(body, false, true);
    }
}

auto atemir::FunctionOp::isDeclaration() -> bool
{
    return isExternal();
}

mlir::LogicalResult atemir::FunctionOp::verifyType()
{
    auto type = getFunctionType();
    if (!isa<atemir::FunctionType>(type))
    {
        return emitOpError(std::string{"requires '"}.append(getFunctionTypeAttrName().str()).append("' attribute of function type"));
    }
    return mlir::success();
}

// Verifies linkage types
// - functions don't have 'common' linkage
// - external functions have 'external' or 'extern_weak' linkage
mlir::LogicalResult atemir::FunctionOp::verify()
{
    if (getLinkage() == GlobalLinkageKind::CommonLinkage)
    {
        return emitOpError() << "functions cannot have '"
                             << stringifyGlobalLinkageKind(GlobalLinkageKind::CommonLinkage) << "' linkage";
    }

    if (isExternal())
    {
        if (getLinkage() != GlobalLinkageKind::ExternalLinkage and
            getLinkage() != GlobalLinkageKind::ExternalWeakLinkage)
        {
            return emitOpError() << "external functions must have '"
                                 << stringifyGlobalLinkageKind(GlobalLinkageKind::ExternalLinkage) << "' or '"
                                 << stringifyGlobalLinkageKind(GlobalLinkageKind::ExternalWeakLinkage) << "' linkage";
        }
        return mlir::success();
    }

    return mlir::success();
}

auto atemir::ConstantOp::inferReturnTypes(mlir::MLIRContext *context, std::optional<mlir::Location> location,
                                          Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferedReturnType)
    -> mlir::LogicalResult
{
    auto type = adaptor.getValueAttr().getType();
    inferedReturnType.push_back(type);
    return mlir::success();
}

auto ensureRegionTerm(mlir::OpAsmParser &parser, mlir::Region &region, mlir::SMLoc errLoc) -> mlir::LogicalResult
{
    mlir::Location eLoc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
    mlir::OpBuilder builder(parser.getBuilder().getContext());

    // Region is empty or properly terminated: nothing to do.
    if (region.empty() or (region.back().mightHaveTerminator() and region.back().getTerminator()))
    {
        return mlir::success();
    }
    // Check for invalid terminator omissions.
    if (!region.hasOneBlock())
    {
        return parser.emitError(errLoc, "multi-block region must not omit terminator");
    }
    if (region.back().empty())
    {
        return parser.emitError(errLoc, "empty region must not omit terminator");
    }

    // Terminator was omited correctly: recreate it.
    region.back().push_back(builder.create<atemir::YieldOp>(eLoc));
    return mlir::success();
}

// True if the region's terminator should be omitted.
auto omitRegionTerm(mlir::Region &r) -> bool
{
    const auto singleNonEmptyBlock = r.hasOneBlock() && !r.back().empty();
    const auto yieldsNothing = [&r]() {
        atemir::YieldOp y = dyn_cast<atemir::YieldOp>(r.back().getTerminator());
        return y && y.getArgs().empty();
    };
    return singleNonEmptyBlock && yieldsNothing();
}

auto atemir::IfOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) -> mlir::ParseResult
{
    // Create the regions for 'then'.
    result.regions.reserve(2);
    mlir::Region *thenRegion = result.addRegion();
    mlir::Region *elseRegion = result.addRegion();

    auto &builder = parser.getBuilder();
    mlir::OpAsmParser::UnresolvedOperand cond;
    mlir::Type boolType = atemir::BooleanType::get(builder.getContext());

    if (parser.parseOperand(cond) or parser.resolveOperand(cond, boolType, result.operands))
    {
        return mlir::failure();
    }

    // Parse the 'then' region.
    auto parseThenLoc = parser.getCurrentLocation();
    if (parser.parseRegion(*thenRegion, /*arguments=*/{},
                           /*argTypes=*/{}))
    {
        return mlir::failure();
    }
    if (ensureRegionTerm(parser, *thenRegion, parseThenLoc).failed())
    {
        return mlir::failure();
    }

    // If we find an 'else' keyword, parse the 'else' region.
    if (!parser.parseOptionalKeyword("else"))
    {
        auto parseElseLoc = parser.getCurrentLocation();
        if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
        {
            return mlir::failure();
        }
        if (ensureRegionTerm(parser, *elseRegion, parseElseLoc).failed())
        {
            return mlir::failure();
        }
    }

    // Parse the optional attribute list.
    if (parser.parseOptionalAttrDict(result.attributes))
    {
        return mlir::failure();
    }
    return mlir::success();
}

auto atemir::IfOp::print(mlir::OpAsmPrinter &p) -> void
{
    p << " " << getCondition() << " ";
    auto &thenRegion = this->getThenRegion();
    p.printRegion(thenRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!omitRegionTerm(thenRegion));

    // Print the 'else' regions if it exists and has a block.
    auto &elseRegion = this->getElseRegion();
    if (!elseRegion.empty())
    {
        p << " else ";
        p.printRegion(elseRegion,
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/!omitRegionTerm(elseRegion));
    }
    p.printOptionalAttrDict(getOperation()->getAttrs());
}

/// Default callback for IfOp builders. Inserts nothing for now.
auto atemir::buildTerminatedBody(mlir::OpBuilder &builder, mlir::Location loc) -> void
{
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
auto atemir::IfOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                       mlir::SmallVectorImpl<mlir::RegionSuccessor> &regions) -> void
{
    // The `then` and the `else` region branch back to the parent operation.
    if (!point.isParent())
    {
        regions.push_back(mlir::RegionSuccessor());
        return;
    }

    // Don't consider the else region if it is empty.
    mlir::Region *elseRegion = &this->getElseRegion();
    if (elseRegion->empty())
    {
        elseRegion = nullptr;
    }

    // Otherwise, the successor is dependent on the condition.
    // bool condition;
    // if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
    //   assert(0 && "not implemented");
    // condition = condAttr.getValue().isOneValue();
    // Add the successor regions using the condition.
    // regions.push_back(RegionSuccessor(condition ? &thenRegion() :
    // elseRegion));
    // return;
    // }

    // If the condition isn't constant, both regions may be executed.
    regions.push_back(mlir::RegionSuccessor(&getThenRegion()));
    // If the else region does not exist, it is not a viable successor.
    if (elseRegion)
    {
        regions.push_back(mlir::RegionSuccessor(elseRegion));
    }
    return;
}

auto atemir::IfOp::build(mlir::OpBuilder &builder, mlir::OperationState &result, mlir::Value cond, bool withElseRegion,
                         mlir::function_ref<void(mlir::OpBuilder &, mlir::Location)> thenBuilder,
                         mlir::function_ref<void(mlir::OpBuilder &, mlir::Location)> elseBuilder) -> void
{
    assert(thenBuilder && "the builder callback for 'then' must be present");

    result.addOperands(cond);

    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Region *thenRegion = result.addRegion();
    builder.createBlock(thenRegion);
    thenBuilder(builder, result.location);

    mlir::Region *elseRegion = result.addRegion();
    if (!withElseRegion)
    {
        return;
    }

    builder.createBlock(elseRegion);
    elseBuilder(builder, result.location);
}

auto atemir::IfOp::verify() -> mlir::LogicalResult
{
    return mlir::success();
}

auto atemir::ConditionOp::getSuccessorRegions(mlir::ArrayRef<mlir::Attribute> operands,
                                              mlir::SmallVectorImpl<mlir::RegionSuccessor> &regions) -> void
{
    // The condition value may be folded to a constant, narrowing
    // down its list of possible successors.

    // Parent is a loop: condition may branch to the body or to the parent op.
    if (auto loopOp = dyn_cast<LoopOpInterface>(getOperation()->getParentOp()))
    {
        regions.emplace_back(&loopOp.getBody(), loopOp.getBody().getArguments());
        regions.emplace_back(loopOp->getResults());
    }
}

auto atemir::ConditionOp::getMutableSuccessorOperands(mlir::RegionBranchPoint point) -> mlir::MutableOperandRange
{
    // No values are yielded to the successor region.
    return mlir::MutableOperandRange(getOperation(), 0, 0);
}

auto atemir::ConditionOp::verify() -> mlir::LogicalResult
{
    if (!isa<LoopOpInterface>(getOperation()->getParentOp()))
        return emitOpError("condition must be within a conditional region");
    return mlir::success();
}

auto atemir::YieldOp::build(mlir::OpBuilder &, mlir::OperationState &) -> void
{
}

auto atemir::DoWhileOp::getSuccessorRegions(::mlir::RegionBranchPoint point,
                                            ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) -> void
{
    LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

auto atemir::DoWhileOp::getLoopRegions() -> ::llvm::SmallVector<mlir::Region *>
{
    return {&getBody()};
}

auto atemir::WhileOp::getSuccessorRegions(::mlir::RegionBranchPoint point,
                                          ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) -> void
{
    LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

auto atemir::WhileOp::getLoopRegions() -> ::llvm::SmallVector<mlir::Region *>
{
    return {&getBody()};
}

auto atemir::ForOp::getSuccessorRegions(::mlir::RegionBranchPoint point,
                                        ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) -> void
{
    LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

auto atemir::ForOp::getLoopRegions() -> ::llvm::SmallVector<mlir::Region *>
{
    return {&getBody()};
}

auto atemir::UnaryOp::verify() -> mlir::LogicalResult
{
    return mlir::success();
}

auto atemir::BinaryOp::verify() -> mlir::LogicalResult
{
    return mlir::success();
}

auto atemir::LoadOp::verify() -> mlir::LogicalResult
{
    return mlir::success();
}

auto atemir::StoreOp::verify() -> mlir::LogicalResult
{
    return mlir::success();
}

mlir::LogicalResult atemir::GlobalOp::verify()
{
    // Verify that the initial value, if present, is either a unit attribute or
    // an attribute CIR supports.
    if (getInitialValue().has_value())
    {
        // if (checkConstantTypes(getOperation(), getSymType(), *getInitialValue()).failed())
        //     return mlir::failure();
    }

    // Verify that the constructor region, if present, has only one block which is
    // not empty.
    auto &initRegion = getInitRegion();
    if (!initRegion.empty())
    {
        if (!initRegion.hasOneBlock())
        {
            return emitError() << "ctor region must have exactly one block.";
        }

        auto &block = initRegion.front();
        if (block.empty())
        {
            return emitError() << "ctor region shall not be empty.";
        }
    }

    // Verify that the destructor region, if present, has only one block which is
    // not empty.
    auto &deinitRegion = getDeinitRegion();
    if (!deinitRegion.empty())
    {
        if (!deinitRegion.hasOneBlock())
        {
            return emitError() << "dtor region must have exactly one block.";
        }

        auto &block = deinitRegion.front();
        if (block.empty())
        {
            return emitError() << "dtor region shall not be empty.";
        }
    }

    if (std::optional<uint64_t> alignAttr = getAlignment())
    {
        uint64_t alignment = alignAttr.value();
        if (!llvm::isPowerOf2_64(alignment))
            return emitError() << "alignment attribute value " << alignment << " is not a power of 2";
    }

    switch (getLinkage())
    {
    case GlobalLinkageKind::InternalLinkage:
    case GlobalLinkageKind::PrivateLinkage:
        if (isPublic())
            return emitError() << "public visibility not allowed with '" << stringifyGlobalLinkageKind(getLinkage())
                               << "' linkage";
        break;
    case GlobalLinkageKind::ExternalLinkage:
    case GlobalLinkageKind::ExternalWeakLinkage:
    case GlobalLinkageKind::LinkOnceODRLinkage:
    case GlobalLinkageKind::LinkOnceAnyLinkage:
    case GlobalLinkageKind::CommonLinkage:
        // FIXME: mlir's concept of visibility gets tricky with LLVM ones,
        // for instance, symbol declarations cannot be "public", so we
        // have to mark them "private" to workaround the symbol verifier.
        if (isPrivate() && !isDeclaration())
            return emitError() << "private visibility not allowed with '" << stringifyGlobalLinkageKind(getLinkage())
                               << "' linkage";
        break;
    default:
        emitError() << stringifyGlobalLinkageKind(getLinkage()) << ": verifier not implemented\n";
        return mlir::failure();
    }

    return mlir::success();
}

void atemir::GlobalOp::build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, mlir::StringRef sym_name,
                             mlir::Type sym_type, bool isConstant, atemir::GlobalLinkageKind linkage,
                             mlir::function_ref<void(mlir::OpBuilder &, mlir::Location)> ctorBuilder,
                             mlir::function_ref<void(mlir::OpBuilder &, mlir::Location)> dtorBuilder)
{
    odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
    odsState.addAttribute(getSymTypeAttrName(odsState.name), ::mlir::TypeAttr::get(sym_type));
    if (isConstant)
        odsState.addAttribute(getConstantAttrName(odsState.name), odsBuilder.getUnitAttr());

    atemir::GlobalLinkageKindAttr linkageAttr = atemir::GlobalLinkageKindAttr::get(odsBuilder.getContext(), linkage);
    odsState.addAttribute(getLinkageAttrName(odsState.name), linkageAttr);

    mlir::Region *initRegion = odsState.addRegion();
    if (ctorBuilder)
    {
        odsBuilder.createBlock(initRegion);
        ctorBuilder(odsBuilder, odsState.location);
    }

    mlir::Region *deinitRegion = odsState.addRegion();
    if (dtorBuilder)
    {
        odsBuilder.createBlock(deinitRegion);
        dtorBuilder(odsBuilder, odsState.location);
    }
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void atemir::GlobalOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                           mlir::SmallVectorImpl<mlir::RegionSuccessor> &regions)
{
    // The `ctor` and `dtor` regions always branch back to the parent operation.
    if (!point.isParent())
    {
        regions.push_back(mlir::RegionSuccessor());
        return;
    }

    // Don't consider the ctor region if it is empty.
    mlir::Region *initRegion = &this->getInitRegion();
    if (initRegion->empty())
        initRegion = nullptr;

    // Don't consider the dtor region if it is empty.
    mlir::Region *deinitRegion = &this->getInitRegion();
    if (deinitRegion->empty())
        deinitRegion = nullptr;

    // If the condition isn't constant, both regions may be executed.
    if (initRegion)
        regions.push_back(mlir::RegionSuccessor(initRegion));
    if (deinitRegion)
        regions.push_back(mlir::RegionSuccessor(deinitRegion));
}