#ifndef ATEM_IR_OPS_TD
#define ATEM_IR_OPS_TD

include "mlir/IR/OpBase.td"

include "AtemIR/Dialect/include/AtemIRDialect.td"
include "AtemIR/Dialect/include/AtemIRTypes.td"
include "AtemIR/Dialect/include/AtemIRAttrs.td"
include "AtemIR/Dialect/include/AtemIROpEnums.td"

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
    let results = (outs AtemIRAnyType:$res);

    let assemblyFormat = "attr-dict $value";
}

def ReturnOp : AtemIROp<"return", [Terminator, ReturnLike]> {
    let summary = "Return operation";
    let description = [{
        `atemir.return` returns from a function with a return value.

        Examples:
        ```mlir
        atemir.function @main() -> !atemir.int<s, 64> {
            %0 = atemir.constant #atemir.int<42>: !atemir.int<s, 64>
            atemir.return %0: !atemir.int<s, 64>
        }
        ```
    }];
    let arguments = (ins Variadic<AtemIRAnyType>:$data);
    let assemblyFormat = "attr-dict ($data^ `:` type($data))?";
}

def FunctionOp : AtemIROp<"function", 
    [
        AutomaticAllocationScope,
        IsolatedFromAbove,
        FunctionOpInterface,
        RegionKindInterface
    ]> {
    let summary = "Function operation";
    let description = [{
        `atemir.function` represents a function in Atem.

        Examples:
        ```mlir
        atemir.function @main() -> !atemir.int<s, 64> {
            %0 = atemir.constant #atemir.int<42>: !atemir.int<s, 64>
            atemir.return %0: !atemir.int<s, 64>
        }
        ```
    }];
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<AtemFunctionType>:$function_type,
        OptionalAttr<StrAttr>:$sym_visibility,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
        "::mlir::StringRef":$name, "::atemir::FunctionType":$type,
        CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs,
        CArg<"::mlir::ArrayRef<::mlir::DictionaryAttr>", "{}">:$argAttrs)
    >];

    let hasCustomAssemblyFormat = true;
    let extraClassDeclaration = [{
        static ::atemir::FunctionOp create(::mlir::Location location, ::mlir::StringRef name, ::atemir::FunctionType type,
                         ::mlir::ArrayRef<::mlir::NamedAttribute> attrs = {});
        static ::atemir::FunctionOp create(::mlir::Location location, ::mlir::StringRef name, ::atemir::FunctionType type,
                         ::mlir::Operation::dialect_attr_range attrs);
        static ::atemir::FunctionOp create(::mlir::Location location, ::mlir::StringRef name, ::atemir::FunctionType type,
                         ::mlir::ArrayRef<::mlir::NamedAttribute> attrs,
                         ::mlir::ArrayRef<::mlir::DictionaryAttr> argAttrs);

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
        Examples:
        ```mlir
        atemir.while {
            atemir.yield
        } do {
            %while_loop_cond = atemir.constant #atemir.bool<true>: !atemir.bool
            atemir.condition(%while_loop_cond)
        }
        ```
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

        Examples:
        ```mlir
        atemir.while {
            atemir.yield
        } do {
            %while_loop_cond = atemir.constant #atemir.bool<true>: !atemir.bool
            atemir.condition(%while_loop_cond)
        }
        ```
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

        Examples:
        ```mlir
        atemir.while {
            atemir.yield
        } do {
            %while_loop_cond = atemir.constant #atemir.bool<true>: !atemir.bool
            atemir.condition(%while_loop_cond)
        }
        ```
    }];
}

def DoWhileOp
    : WhileOpBase<"do"> {
    let regions = (region SizedRegion<1>:$body, MinSizedRegion<1>:$cond);
    let assemblyFormat = "$body `while` $cond attr-dict";

    let description = [{
        The `atemir.do` operation represents a do-while loop in Atem. Identical
        to `atemir.while` but the condition is evaluated after the body.

        Examples:
        ```mlir
        atemir.do {
            %do_while_loop_cond = atemir.constant #atemir.bool<true>: !atemir.bool
            atemir.condition(%do_while_loop_cond)
        } while {
            atemir.yield
        }
        ```
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

        Example:
        ```mlir
        atemir.cfor : cond {
            %cfor_loop_cond = atemir.constant #atemir.bool<true>: !atemir.bool
            atemir.condition(%cfor_loop_cond)
        } body {
            atemir.yield
        } step {
            atemir.yield
        }
        ```
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

def UnaryOp
    : AtemIROp<"unary", [Pure, SameOperandsAndResultType]> {
    let summary = "Unary operations";
    let description = [{
        `atemir.unary` performs the unary operation according to the
        specified operation kind.

        It requires one input operand and has one result, both types
        should be the same.

        Examples:
        ```mlir
        %0 = atemir.unary neg %1 : !atemir.int<s, 64> -> !atemir.int<s, 64>
        ```
    }];

    let results = (outs AtemIRAnyType:$result);
    let arguments = (ins Arg<UnaryOpKind, "UnaryOpKind">:$kind, Arg<AtemIRAnyType>:$input);

    let assemblyFormat = [{
        $kind $input `:` type($input) `->` type($result) attr-dict
    }];

    let hasVerifier = 1;
}

def BinaryOp
    : AtemIROp<"binary", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
    let summary = "Binary operations";
    let description = [{
        `atemir.binary` performs the binary operation according to the
        specified operation kind.

        It requires two input operand and has one result, all types
        should be the same.

        Examples:
        ```mlir
        %0 = atemir.binary add %1 %2 : !atemir.int<s, 64> -> !atemir.int<s, 64>
        ```
    }];

    let results = (outs AtemIRAnyType:$result);
    let arguments = (ins 
        Arg<BinaryOpKind, "BinaryOpKind">:$kind,
        AtemIRAnyType:$lhs,
        AtemIRAnyType:$rhs,
        UnitAttr:$no_unsigned_wrap,
        UnitAttr:$no_signed_wrap
    );

    let assemblyFormat = [{
        $kind $lhs $rhs 
        (`nsw` $no_signed_wrap^)?
        (`nuw` $no_unsigned_wrap^)?
        `:` type($lhs) `->` type($result) attr-dict
    }];

    let hasVerifier = 1;
}

def CompareOp
    : AtemIROp<"compare", [Pure, SameTypeOperands]> {
    let summary = "Compare two values and produce a boolean result";
    let description = [{
        `atemir.cmp` compares two input operands of the same type and produces a
        `atemir.bool` result. The kinds of comparison available are:
        - `lt`: Lesser Than
        - `le`: 
        - `gt`: Greater Than
        - `ge`: 
        - `eq`: Equal
        - `ne`: Not Equal

        Examples:
        ```mlir
        %0 = atemir.cmp gt %1 %2 : !atemir.int<s, 64>
        ```
    }];

    let arguments = (ins Arg<CompareOpKind, "compare kind">:$kind, AtemIRAnyType:$lhs, AtemIRAnyType:$rhs);
    let results = (outs BooleanType:$result);

    let assemblyFormat = [{
        $kind $lhs $rhs `:` type($lhs) attr-dict
    }];

    let hasVerifier = 0;
}

class AllocateTypesMatchWith<string summary, string lhsArg, string rhsArg, string transform,
    string comparator = "std::equal_to<>()"> 
    : PredOpTrait<summary, CPred<
        comparator # "(" #
        !subst("$_self", "$" # lhsArg # ".getType()", transform) #
        ", $" # rhsArg # ")"
    >> {
    string lhs = lhsArg;
    string rhs = rhsArg;
    string transformer = transform;
}

def AllocateOp
    : AtemIROp<"allocate", [
        AllocateTypesMatchWith<
            "'allocateType' matches pointee type of 'addr'",
            "addr", "allocateType",
            "::mlir::cast<::atemir::PointerType>($_self).getPointee()"
        >
    ]> {
    let summary = "Define a local variable in Atem";
    let description = [{
        `atemir.allocate` operation defines a local variable in scope.

        The presence `init` attribute indicates that the local variable represented
        by this operation was originally initialized in Atem source code. In such
        cases, the first use contains the initialization.

        The result type is a pointer to the input's type.

        Examples:
        ```mlir
        //count: var int32 = 3
        %0 = atemir.allocate !atemir.int<s, 32>, !atemir.ptr<!atemir.int<s, 32>>, ["count", init] {alignment = 4 : !atemir.int<s, 64>}
        ```
    }];

    let arguments = (ins
        TypeAttr:$allocateType,
        StrAttr:$name,
        UnitAttr:$init,
        ConfinedAttr<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment
    );
    let results = (outs Res<PointerType, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "::mlir::Type":$addr, "::mlir::Type":$allocaType,
                    "::mlir::StringRef":$name,
                    "::mlir::IntegerAttr":$alignment)>
    ];

    let extraClassDeclaration = [{
        // Whether the alloca input type is a pointer.
        bool isPointerType() { return ::mlir::isa<::atemir::PointerType>(getAllocateType()); }
    }];

  // FIXME: we should not be printing `cir.ptr` below, that should come
  // from the pointer type directly.
    let assemblyFormat = [{
        $allocateType `->` type($addr) `,`
        `[` $name
        (`,` `init` $init^)?
        `]`
        attr-dict
    }];

    let hasVerifier = 0;
}

def LoadOp
    : AtemIROp<"load", [
        TypesMatchWith<
            "type of 'result' matches pointee type of 'addr'",
            "addr", "result",
            "::mlir::cast<::atemir::PointerType>($_self).getPointee()"
        >
    ]> {
    let summary = "Load value from specified memory address";
    let description = [{
        `atemir.load` reads a value given an address backed up by a
        `atemir.ptr` type. A unit attribute `deref` can be used to mark
        the resulting value as used by another operation to dereference
        a pointer. A unit attribute `volatile` can be used to indicate a
        volatile loading. Load can be marked atomic by using 
        `atomic(<mem_order>)`.

        Examples:
        ```mlir
        //Read from local variable, address stored in %0.
        %1 = atemir.load %0 : !atemir.ptr<!atemir.int<s, 64>> -> !atemir.int<s, 64>

        //Load address from memory at address %0. %3 is used by at least one operation
        //that dereferences a pointer.
        %3 = atemir.load deref %0 : !atemir.ptr<!atemir.ptr<!atemir.int<s, 64>>> -> !atemir.ptr<!atemir.int<s, 64>>

        /Perform a volatile load from address in %0
        %4 = atemir.load volatile %0 : !atemir.ptr<!atemir.int<s, 64>> -> !atemir.int<s, 64>
        ```
    }];

    let arguments = (ins 
        Arg<PointerType, "the address to load from", [MemRead]>:$addr,
        UnitAttr:$isDeref,
        UnitAttr:$isVolatile,
        OptionalAttr<MemoryOrderKind>:$memoryOrder
    );
    let results = (outs AtemIRAnyType:$result);

    let assemblyFormat = [{
        (`deref` $isDeref^)?
        (`volatile` $isVolatile^)?
        (`atomic` `(` $memoryOrder^ `)`)?
        $addr `:` type($addr) `->` type($result) attr-dict
    }];
    
    let hasVerifier = 1;
}

def StoreOp
    : AtemIROp<"store", [
        TypesMatchWith<
            "type of 'result' matches pointee type of 'addr'",
            "addr", "result",
            "::mlir::cast<::atemir::PointerType>($_self).getPointee()"
        >
    ]> {
    let summary = "Store value to specified memory address";
    let description = [{
        `atemir.store` stores a value to the specified memory address. A
        unit attribute `volatile` can be used to indicate a volatile store.
        Store operation can be marked atomic by using `atomic(<mem_order>)`.

        Examples:
        ```mlir
        //Store a function argument to local storage, address in %0.
        atemir.store %arg0 : !atemir.int<s, 64>, %0 : !atemir.ptr<!atemir.int<s, 64>>

        //Perform a volatile store into memory location at the address in %0
        atemir.store volatile %arg0 : !atemir.int<s, 64>, %0 : !atemir.ptr<!atemir.int<s, 64>>
        ```
    }];

    let arguments = (ins 
        AtemIRAnyType:$value,
        Arg<PointerType, "the address to store the value", [MemRead]>:$addr,
        UnitAttr:$isVolatile,
        OptionalAttr<MemoryOrderKind>:$memoryOrder
    );
    let results = (outs AtemIRAnyType:$result);

    let assemblyFormat = [{
        (`volatile` $isVolatile^)?
        (`atomic` `(` $memoryOrder^ `)`)?
        $value `:` type($value) `,` $addr `:` type($addr) attr-dict
    }];
    
    let hasVerifier = 1;
}

def GlobalOp
    : AtemIROp<"global", [
        Symbol,
        DeclareOpInterfaceMethods<RegionBranchOpInterface>,
        NoRegionArguments
    ]> {
    let summary = "Defines a global variable in Atem";
    let description = [{
        `atemir.global` operation defines a named global variable in atem.

        The backing memory for the variable is allocated statically and is 
        described by the type of the variable.

        The operation may has a `initial_value`. 
        
        The global variable can also be marked constant using the `constant`
        unit attribute. Writing to such constant global variables is prohibited
        and will be diagnosed by the compiler.

        The `linkage` attribute is used to provide LLVM linkage information. 
        Symbol visibility in `sym_visibility` is defined in terms of MLIR's
        visibility and verified to be in accordance to `linkage`.

        Examples:
        ```mlir
        atem.global public constant @c : !atemir.int<s, 64> = 4;
        ```
    }];
    
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<StrAttr>:$sym_visibility,
        TypeAttr:$sym_type,
        Arg<GlobalLinkageKind, "linkage type">:$linkage,
        OptionalAttr<TLSModel>:$tls_model,
        OptionalAttr<AnyAttr>:$initial_value,
        UnitAttr:$constant,
        ConfinedAttr<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment
    );
    let regions = (region AnyRegion:$initRegion, AnyRegion:$deinitRegion);
    let assemblyFormat = [{
        ($sym_visibility^)?
       (`constant` $constant^)?
       $linkage
       ($tls_model^)?
       $sym_name
       custom<GlobalOpTypeAndInitialValue>($sym_type, $initial_value, $initRegion, $deinitRegion)
       attr-dict
    }];

    let extraClassDeclaration = [{
        bool isDeclaration() {
            return !getInitialValue() && getInitRegion().empty() && getDeinitRegion().empty();
        }
        bool hasInitializer() { return !isDeclaration(); }
        bool hasAvailableExternallyLinkage() {
            return atemir::isAvailableExternallyLinkage(getLinkage());
        }
        bool isDeclarationForLinker() {
            if (hasAvailableExternallyLinkage())
                return true;

            return isDeclaration();
        }

        /// Whether the definition of this global may be replaced at link time.
        bool isWeakForLinker() { return atemir::isWeakForLinker(getLinkage()); }
    }];

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
        // MLIR's default visibility is public.
        "::mlir::StringRef":$sym_name,
        "::mlir::Type":$sym_type,
        CArg<"bool", "false">:$isConstant,
        // CIR defaults to external linkage.
        CArg<"atemir::GlobalLinkageKind",
                "atemir::GlobalLinkageKind::ExternalLinkage">:$linkage,
        CArg<"::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>",
            "nullptr">:$ctorBuilder,
        CArg<"::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>",
            "nullptr">:$dtorBuilder)>
    ];

    let skipDefaultBuilders = 1;
    let hasVerifier = 1;
}

#endif  //ATEM_IR_OPS_TD