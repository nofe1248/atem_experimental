#ifndef ATEM_IR_OPS_TD
#define ATEM_IR_OPS_TD

include "mlir/IR/OpBase.td"

include "AtemIR/Dialect/include/AtemIRDialect.td"
include "AtemIR/Dialect/include/AtemIRTypes.td"
include "AtemIR/Dialect/include/AtemIRAttrs.td"

include "AtemIR/Interfaces/include/AtemIRLoopOpInterface.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpAsmInterface.td"

class AtemIROp<string mnemonic, list<Trait> traits = []> :
    Op<AtemIRDialect, mnemonic, traits>;

def ConstantOp
    : AtemIROp<"constant", 
    [ConstantLike, Pure, 
    InferTypeOpAdaptor]> {
    let summary = "Constant operation";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:
        ```mlir
        %0 = atemir.constant(42: i32): atemir.Int<32>
        ```
    }];

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs AnyType:$res);

    let assemblyFormat = "attr-dict $value";
}

def ReturnOp : AtemIROp<"return", [Terminator, ReturnLike]> {
    let summary = "Return operation";
    let arguments = (ins AnyType:$data);
    let assemblyFormat = "$data attr-dict `:` type($data)";
}

def FunctionOp : AtemIROp<"function", [
    IsolatedFromAbove,
    FunctionOpInterface,
    RegionKindInterface]> {
    let summary = "Function operation";
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    dag regions = (region AnyRegion:$body);
    let hasCustomAssemblyFormat = true;
    let extraClassDeclaration = [{
        mlir::Region * getCallableRegion() {return &getBody();}

        // Method of CallableOpInterface
        llvm::ArrayRef<mlir::Type> getArgumentTypes() {return getFunctionType().getInputs();}
        llvm::ArrayRef<mlir::Type> getResultTypes() {return getFunctionType().getResults();}

        // Method of RegionKindInterface
        static mlir::RegionKind getRegionKind(unsigned idx) { return mlir::RegionKind::SSACFG; }
    }];
}

def CallOp : AtemIROp<"call", [CallOpInterface]> {
    let summary = "Function Call operation";
    let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$arg_operands);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$callee `(` $arg_operands `)` attr-dict `:` functional-type($arg_operands, results)";
    let extraClassDeclaration = [{
        mlir::CallInterfaceCallable getCallableForCallee() { return getCalleeAttr(); }
        void setCalleeFromCallable(mlir::CallInterfaceCallable callee) { setCalleeAttr(callee.get<mlir::SymbolRefAttr>()); }
    }];
}

def IfOp
    : AtemIROp<"if",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    RecursivelySpeculatable, AutomaticAllocationScope, NoRegionArguments]> {
    let summary = "The if-then-else operation";
    let description = [{
        The `atemir.if` operation represents an if-then-else expression for
        conditionally executing two regions of code and yielding their
        values. The operand of this operation is a `atemir.bool` type.

        Examples:

        ```mlir
        atemir.if %cond {
            ...
        } else {
            ...
        }
        ```
    }];

    let arguments = (ins BooleanType:$condition);
    let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let skipDefaultBuilders = 1;
    let builders = [
    OpBuilder<(ins "::mlir::Value":$cond, "bool":$withElseRegion,
        CArg<"::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>",
            "::atemir::buildTerminatedBody">:$thenBuilder,
        CArg<"::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>",
            "nullptr">:$elseBuilder)>
    ];
}

def YieldOp
    : AtemIROp<"yield", [ReturnLike, Terminator]> {
    let summary = "Yielding the control from current block to its parent block";
    let description = [{

    }];

    let arguments = (ins Variadic<AtemIRAnyType>:$args);
    let assemblyFormat = "($args^ `:` type($args))? attr-dict";
    let builders = [
        OpBuilder<(ins), [{}]>,
    ];
}

def ConditionOp
    : AtemIROp<"condition", [Terminator, 
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface, ["getSuccessorRegions"]>
    ]> {
    let summary = "Loop continuation condition operation";
    let description = [{
        The `atemir.condition` operation terminates conditional regions. It
        taks a single `atemir.bool` operand and, depending on its value, may
        branch to different regions.
        This operation can be only used in `cir.while` and `cir.do` currently,
        but the set of usable operations is likely to be expanded in the future.
    }];

    let arguments = (ins BooleanType:$cond);
    let assemblyFormat = " `(` $cond `)` attr-dict ";
    let hasVerifier = 1;
}

class WhileOpBase<string mnemonic>
    : AtemIROp<mnemonic, [LoopOpInterface, NoRegionArguments]> {
    defvar isWhile = !eq(mnemonic, "while");
    let summary = "The " # !if(isWhile, "while", "do-while") # " loop operation";
    let builders = [
    OpBuilder<(ins "::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$condBuilder,
                    "::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$bodyBuilder), [{
            ::mlir::OpBuilder::InsertionGuard guard($_builder);
            $_builder.createBlock($_state.addRegion());
        }] # !if(isWhile, [{
            condBuilder($_builder, $_state.location);
            $_builder.createBlock($_state.addRegion());
            bodyBuilder($_builder, $_state.location);
        }], [{
            bodyBuilder($_builder, $_state.location);
            $_builder.createBlock($_state.addRegion());
            condBuilder($_builder, $_state.location);
        }])>
    ];
}

def WhileOp
    : WhileOpBase<"while"> {
    let regions = (region SizedRegion<1>:$cond, MinSizedRegion<1>:$body);
    let assemblyFormat = "$cond `do` $body attr-dict";

    let description = [{
        The `atemir.while` operation represents a while loop in Atem. It consists
        of two regions:

        - `cond`: single block region with the loop's condition. Should be terminated
        with a `atemir.condition` operation.
        - `body`: contains the loop body and an arbitrary number of blocks.
    }];
}

def DoWhileOp
    : WhileOpBase<"do"> {
    let regions = (region SizedRegion<1>:$body, MinSizedRegion<1>:$cond);
    let assemblyFormat = "$body `while` $cond attr-dict";

    let description = [{
        The `atemir.do` operation represents a do-while loop in Atem. Identical
        to `atemir.while` but the condition is evaluated after the body.
    }];
}

def ForOp
    : AtemIROp<"cfor", [LoopOpInterface, NoRegionArguments]> {
    let summary = "Atem C-style for loop";
    let description = [{
        Represents a C/C++ style for loop. It consists of three regions:

        - `cond`: single block region with the loop's condition. Should be terminated
        with a `atemir.condition` operation.
        - `body`: contains the loop body and an arbitrary number of blocks.
        - `step`: single block region with the loop's step.
    }];

    let regions = (region
        SizedRegion<1>:$cond,
        MinSizedRegion<1>:$body,
        SizedRegion<1>:$step
    );
    let assemblyFormat = [{
        `:` `cond` $cond
        `body` $body
        `step` $step
        attr-dict
    }];

    let builders = [
        OpBuilder<(ins
            "::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$condBuilder,
            "::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$bodyBuilder,
            "::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>":$stepBuilder
        ), [{
            ::mlir::OpBuilder::InsertionGuard guard($_builder);

            // Build condition region.
            $_builder.createBlock($_state.addRegion());
            condBuilder($_builder, $_state.location);

            // Build body region.
            $_builder.createBlock($_state.addRegion());
            bodyBuilder($_builder, $_state.location);

            // Build step region.
            $_builder.createBlock($_state.addRegion());
            stepBuilder($_builder, $_state.location);
        }]>
    ];

    let extraClassDeclaration = [{
        auto maybeGetStep() -> ::mlir::Region* { return &this->getStep(); }
        auto getRegionsInExecutionOrder() -> ::llvm::SmallVector<::mlir::Region *> {
            return ::llvm::SmallVector<::mlir::Region *, 3>{
                &this->getCond(), 
                &this->getBody(), 
                &this->getStep()
            };
        }
    }];
}

#endif  //ATEM_IR_OPS_TD