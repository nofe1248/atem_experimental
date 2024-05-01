#ifndef ATEM_IR_OPS_TD
#define ATEM_IR_OPS_TD

include "mlir/IR/OpBase.td"

include "AtemIR/include/AtemIRDialect.td"
include "AtemIR/include/AtemIRTypes.td"
include "AtemIR/include/AtemIRAttrs.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

class AtemIROp<string mnemonic, list<Trait> traits = []> :
    Op<AtemIRDialect, mnemonic, traits>;

def ConstantOp : AtemIROp<"constant", [ConstantLike, Pure]> {
    let summary = "Constant operation";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:
        ```mlir
        %0 = atemir.constant(42: i32): i32
        ```
    }];

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs AtemIRAnyType:$res);

    let assemblyFormat = [{
        `(` custom<ConstantValue>($value) `)` attr-dict `:` type($res)
    }];
}

#endif  //ATEM_IR_OPS_TD