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


static auto parseConstantValue(OpAsmParser &parser, mlir::Attribute &valueAttr) -> ParseResult;
static auto printConstantValue(OpAsmPrinter &p, atemir::ConstantOp op, Attribute value) -> void;

#define GET_OP_CLASSES
#include "Dialect/include/AtemIR.cpp.inc"

using namespace mlir;
using namespace atemir;

auto AtemIRDialect::registerOperations() -> void
{
    // Register tablegen'd operations.
    addOperations<
        #define GET_OP_LIST
        #include "Dialect/include/AtemIR.cpp.inc"
    >();
}

static auto parseConstantValue(OpAsmParser &parser, mlir::Attribute &valueAttr) -> ParseResult
{
    NamedAttrList attr;
    return parser.parseAttribute(valueAttr, "value", attr);
}

static auto printConstant(OpAsmPrinter &p, Attribute value) -> void {
    p.printAttribute(value);
}

static auto printConstantValue(OpAsmPrinter &p, atemir::ConstantOp op, Attribute value) -> void
{
    printConstant(p, value);
}

auto FunctionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) -> mlir::ParseResult
{
    auto buildFuncType = [](auto & builder, auto argTypes, auto results, auto, auto) {
        return builder.getFunctionType(argTypes, results);
    };
    return function_interface_impl::parseFunctionOp(
      parser, result, false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name)
    );
}

auto FunctionOp::print(mlir::OpAsmPrinter &p) -> void
{
    // Dispatch to the FunctionOpInterface provided utility method that prints the
    // function operation.
    mlir::function_interface_impl::printFunctionOp(
        p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
}

auto ConstantOp::inferReturnTypes(mlir::MLIRContext *context, std::optional<mlir::Location> location,
                                        Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferedReturnType)
    -> mlir::LogicalResult
{
    auto type = adaptor.getValueAttr().getType();
    inferedReturnType.push_back(type);
    return mlir::success();
}

auto ensureRegionTerm(OpAsmParser &parser, Region &region,
                               SMLoc errLoc) -> LogicalResult
{
    Location eLoc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
    OpBuilder builder(parser.getBuilder().getContext());

    // Region is empty or properly terminated: nothing to do.
    if (region.empty() or
        (region.back().mightHaveTerminator() and region.back().getTerminator()))
    {
        return success();
    }
    // Check for invalid terminator omissions.
    if (!region.hasOneBlock())
    {
        return parser.emitError(errLoc,
                                "multi-block region must not omit terminator");
    }
    if (region.back().empty())
    {
        return parser.emitError(errLoc, "empty region must not omit terminator");
    }

    // Terminator was omited correctly: recreate it.
    region.back().push_back(builder.create<YieldOp>(eLoc));
    return success();
}

// True if the region's terminator should be omitted.
auto omitRegionTerm(mlir::Region &r) -> bool {
    const auto singleNonEmptyBlock = r.hasOneBlock() && !r.back().empty();
    const auto yieldsNothing = [&r]() {
        YieldOp y = dyn_cast<YieldOp>(r.back().getTerminator());
        return y && y.getArgs().empty();
    };
    return singleNonEmptyBlock && yieldsNothing();
}

auto ::atemir::IfOp::parse(OpAsmParser &parser, OperationState &result) -> ParseResult
{
      // Create the regions for 'then'.
      result.regions.reserve(2);
      Region *thenRegion = result.addRegion();
      Region *elseRegion = result.addRegion();

      auto &builder = parser.getBuilder();
      OpAsmParser::UnresolvedOperand cond;
      Type boolType = ::atemir::BooleanType::get(builder.getContext());

      if (parser.parseOperand(cond) or parser.resolveOperand(cond, boolType, result.operands))
      {
          return failure();
      }

      // Parse the 'then' region.
      auto parseThenLoc = parser.getCurrentLocation();
      if (parser.parseRegion(*thenRegion, /*arguments=*/{},
                             /*argTypes=*/{}))
      {
          return failure();
      }
      if (ensureRegionTerm(parser, *thenRegion, parseThenLoc).failed())
      {
          return failure();
      }

      // If we find an 'else' keyword, parse the 'else' region.
      if (!parser.parseOptionalKeyword("else"))
      {
            auto parseElseLoc = parser.getCurrentLocation();
            if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
            {
                return failure();
            }
            if (ensureRegionTerm(parser, *elseRegion, parseElseLoc).failed())
            {
                return failure();
            }
      }

      // Parse the optional attribute list.
      if (parser.parseOptionalAttrDict(result.attributes))
      {
          return failure();
      }
      return success();
}

auto ::atemir::IfOp::print(OpAsmPrinter &p) -> void
{
      p << " " << getCondition() << " ";
      auto &thenRegion = this->getThenRegion();
      p.printRegion(thenRegion,
                    /*printEntryBlockArgs=*/false,
                    /*printBlockTerminators=*/!omitRegionTerm(thenRegion));

      // Print the 'else' regions if it exists and has a block.
      auto &elseRegion = this->getElseRegion();
      if (!elseRegion.empty()) {
            p << " else ";
            p.printRegion(elseRegion,
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/!omitRegionTerm(elseRegion));
      }
      p.printOptionalAttrDict(getOperation()->getAttrs());
}

/// Default callback for IfOp builders. Inserts nothing for now.
auto ::atemir::buildTerminatedBody(OpBuilder &builder, Location loc) -> void {}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
auto ::atemir::IfOp::getSuccessorRegions(mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions)
    -> void
{
      // The `then` and the `else` region branch back to the parent operation.
      if (!point.isParent())
      {
            regions.push_back(RegionSuccessor());
            return;
      }

      // Don't consider the else region if it is empty.
      Region *elseRegion = &this->getElseRegion();
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
      regions.push_back(RegionSuccessor(&getThenRegion()));
      // If the else region does not exist, it is not a viable successor.
      if (elseRegion)
      {
          regions.push_back(RegionSuccessor(elseRegion));
      }
      return;
}

auto ::atemir::IfOp::build(OpBuilder &builder, OperationState &result, Value cond, bool withElseRegion,
                                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                                 function_ref<void(OpBuilder &, Location)> elseBuilder) -> void
{
      assert(thenBuilder && "the builder callback for 'then' must be present");

      result.addOperands(cond);

      OpBuilder::InsertionGuard guard(builder);
      Region *thenRegion = result.addRegion();
      builder.createBlock(thenRegion);
      thenBuilder(builder, result.location);

      Region *elseRegion = result.addRegion();
      if (!withElseRegion)
      {
          return;
      }

      builder.createBlock(elseRegion);
      elseBuilder(builder, result.location);
}

auto ::atemir::IfOp::verify() -> LogicalResult
{
    return success();
}

auto ::atemir::ConditionOp::getSuccessorRegions(ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions)
    -> void
{
    // The condition value may be folded to a constant, narrowing
    // down its list of possible successors.

    // Parent is a loop: condition may branch to the body or to the parent op.
    if (auto loopOp = dyn_cast<LoopOpInterface>(getOperation()->getParentOp())) {
        regions.emplace_back(&loopOp.getBody(), loopOp.getBody().getArguments());
        regions.emplace_back(loopOp->getResults());
    }
}

auto ::atemir::ConditionOp::getMutableSuccessorOperands(RegionBranchPoint point) -> MutableOperandRange
{
    // No values are yielded to the successor region.
    return MutableOperandRange(getOperation(), 0, 0);
}

auto ::atemir::ConditionOp::verify() -> LogicalResult
{
    if (!isa<LoopOpInterface>(getOperation()->getParentOp()))
        return emitOpError("condition must be within a conditional region");
    return success();
}

auto ::atemir::YieldOp::build(mlir::OpBuilder&, mlir::OperationState&) -> void
{

}

auto ::atemir::DoWhileOp::getSuccessorRegions(::mlir::RegionBranchPoint point,
                                              ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) -> void
{
    LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

auto ::atemir::DoWhileOp::getLoopRegions() -> ::llvm::SmallVector<Region *> {
    return {&getBody()};
}

auto ::atemir::WhileOp::getSuccessorRegions(::mlir::RegionBranchPoint point,
                                            ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) -> void
{
    LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

auto ::atemir::WhileOp::getLoopRegions() -> ::llvm::SmallVector<Region *>
{
    return {&getBody()};
}

auto ::atemir::ForOp::getSuccessorRegions(::mlir::RegionBranchPoint point,
                                            ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions) -> void
{
    LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

auto ::atemir::ForOp::getLoopRegions() -> ::llvm::SmallVector<Region *>
{
    return {&getBody()};
}

auto ::atemir::UnaryOp::verify() -> LogicalResult
{
    return success();
}

auto ::atemir::BinaryOp::verify() -> LogicalResult
{
    return success();
}