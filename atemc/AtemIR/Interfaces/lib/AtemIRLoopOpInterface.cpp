#include "AtemIR/Interfaces/include/AtemIRLoopOpInterface.h"

#include "AtemIR/Dialect/include/AtemIRDialect.h"

#include "Interfaces/include/AtemIRLoopOpInterface.cpp.inc"

namespace atemir
{
    auto LoopOpInterface::getLoopOpSuccessorRegions(
            LoopOpInterface op,
            mlir::RegionBranchPoint point,
            mlir::SmallVectorImpl<mlir::RegionSuccessor> &regions
        ) -> void
    {
        assert(point.isParent() || point.getRegionOrNull());

        // Branching to first region: go to condition or body (do-while).
        if (point.isParent()) {
            regions.emplace_back(&op.getEntry(), op.getEntry().getArguments());
        }
        // Branching from condition: go to body or exit.
        else if (&op.getCond() == point.getRegionOrNull()) {
            regions.emplace_back(mlir::RegionSuccessor(op->getResults()));
            regions.emplace_back(&op.getBody(), op.getBody().getArguments());
        }
        // Branching from body: go to step (for) or condition.
        else if (&op.getBody() == point.getRegionOrNull()) {
            // FIXME(cir): Should we consider break/continue statements here?
            auto *afterBody = (op.maybeGetStep() ? op.maybeGetStep() : &op.getCond());
            regions.emplace_back(afterBody, afterBody->getArguments());
        }
        // Branching from step: go to condition.
        else if (op.maybeGetStep() == point.getRegionOrNull()) {
            regions.emplace_back(&op.getCond(), op.getCond().getArguments());
        } else {
            llvm_unreachable("unexpected branch origin");
        }
    }

    auto detail::verifyLoopOpInterface(mlir::Operation *Op) -> mlir::LogicalResult
    {
        return mlir::success();
    }
}
