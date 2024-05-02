#ifndef ATEM_IR_LOOP_OP_INTERFACE_TD
#define ATEM_IR_LOOP_OP_INTERFACE_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

def LoopOpInterface
    : OpInterface<"LoopOpInterface", [
        DeclareOpInterfaceMethods<RegionBranchOpInterface>,
        DeclareOpInterfaceMethods<LoopLikeOpInterface>
    ]> {
    let description = [{
        Contains helper functionsn to query properties and perform transformations
        on an Atem IR loop operation.
    }];
    let cppNamespace = "::atemir";

    let methods = [
        InterfaceMethod<[{
            Returns the loop's conditional region.
        }],
        /*retTy=*/"::mlir::Region &",
        /*methodName=*/"getCond"
        >,
        InterfaceMethod<[{
            Returns the loop's body region.
        }],
        /*retTy=*/"::mlir::Region &",
        /*methodName=*/"getBody"
        >,
        InterfaceMethod<[{
            Returns a pointer to the loop's step region or nullptr.
        }],
        /*retTy=*/"::mlir::Region *",
        /*methodName=*/"maybeGetStep",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return nullptr;"
        >,
        InterfaceMethod<[{
            Returns the first region to be executed in the loop.
        }],
        /*retTy=*/"::mlir::Region &",
        /*methodName=*/"getEntry",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return $_op.getCond();"
        >,
        InterfaceMethod<[{
            Returns a list of regions in order of execution.
        }],
        /*retTy=*/"::llvm::SmallVector<::mlir::Region *>",
        /*methodName=*/"getRegionsInExecutionOrder",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
            return ::llvm::SmallVector<::mlir::Region *, 2>{&$_op.getRegion(0), &$_op.getRegion(1)};
        }]
        >,
        InterfaceMethod<[{
            Recursively walks the body of the loop in pre-order while skipping
            nested loops and executing a callback on every other operation.
        }],
        /*retTy=*/"::mlir::WalkResult",
        /*methodName=*/"walkBodySkippingNestedLoops",
        /*args=*/(ins "::llvm::function_ref<void (::mlir::Operation *)>":$callback),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
            return $_op.getBody().template walk<::mlir::WalkOrder::PreOrder>([&](::mlir::Operation *op) {
            if (::mlir::isa<LoopOpInterface>(op))
                return ::mlir::WalkResult::skip();
            callback(op);
            return ::mlir::WalkResult::advance();
            });
        }]
        >
    ];

    let extraClassDeclaration = [{
        /// Generic method to retrieve the successors of a LoopOpInterface operation.
        static auto getLoopOpSuccessorRegions(
            ::atemir::LoopOpInterface op, ::mlir::RegionBranchPoint point,
            ::mlir::SmallVectorImpl<::mlir::RegionSuccessor> &regions) -> void;
    }];

    let verify = [{
        /// Verify invariants of the LoopOpInterface.
        return detail::verifyLoopOpInterface($_op);
    }];
}

#endif //ATEM_IR_LOOP_OP_INTERFACE_TD