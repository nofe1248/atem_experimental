#ifndef ATEM_IR_TYPES_TD
#define ATEM_IR_TYPES_TD

include "AtemIR/include/AtemIRDialect.td"

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class AtemIRType<string name, string typeMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<AtemIRDialect, name, traits, baseCppClass> {
    let mnemonic = typeMnemonic;
}

def AtemIRIntegerType : 
    AtemIRType<"Integer", "Int"> {

    let summary = "AtemIR integer type with arbitrary precision up to a fixed limit";
    let description = [{
        Integer types have a designated bit width.
    }];
    
    let parameters = (ins "unsigned":$width);

    let assemblyFormat = "`<` $width `>`";
}

def AtemIRBooleanType :
    AtemIRType<"Boolean", "Bool",
             [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

    let summary = "AtemIR bool type";
    let description = [{
        `atemir.bool` represents Atem bool type.
    }];

    let assemblyFormat = "";
}

def AtemIRAnyType : AnyTypeOf<[
    AtemIRIntegerType, AtemIRBooleanType
]>;

#endif //ATEM_IR_TYPES_TD