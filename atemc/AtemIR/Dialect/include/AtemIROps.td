#ifndef ATEM_IR_OPS_TD
#define ATEM_IR_OPS_TD

include "mlir/IR/OpBase.td"

include "AtemIR/Dialect/include/AtemIRDialect.td"
include "AtemIR/Dialect/include/AtemIRTypes.td"
include "AtemIR/Dialect/include/AtemIRAttrs.td"

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

#endif  //ATEM_IR_OPS_TD