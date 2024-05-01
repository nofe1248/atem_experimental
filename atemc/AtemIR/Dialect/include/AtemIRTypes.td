#ifndef ATEM_IR_TYPES_TD
#define ATEM_IR_TYPES_TD

include "AtemIR/Dialect/include/AtemIRDialect.td"

include "AtemIR/Interfaces/AtemIRFPTypeInterface.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

class AtemIRType<string name, string typeMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<AtemIRDialect, name, traits, baseCppClass> {
    let mnemonic = typeMnemonic;
}

def AtemIRIntegerType : 
    AtemIRType<"Integer", "int", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

    let summary = "AtemIR integer type with arbitrary precision up to a fixed limit";
    let description = [{
        Integer types have a designated bit width.
    }];
    
    let parameters = (ins "unsigned":$width, "bool":$isSigned);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        /// Return true if this is a signed integer type.
        auto isSigned() const -> bool { return getIsSigned(); }
        /// Return true if this is an unsigned integer type.
        auto isUnsigned() const -> bool { return !getIsSigned(); }
    }];
}

def AtemIRBooleanType :
    AtemIRType<"Boolean", "bool", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

    let summary = "AtemIR bool type";
    let description = [{
        `atemir.bool` represents Atem bool type.
    }];

    let hasCustomAssemblyFormat = 1;
}

class AtemIRFloatingPointType<string name, string mnemonic>
    : AtemIRType<name, mnemonic, 
    [
        DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
        DeclareTypeInterfaceMethods<AtemIRFPTypeInterface>,
    ]> {}

def AtemIRFloat16Type
    : AtemIRFloatingPointType<"Float16", "float16"> {
    let summary = "AtemIR float16 type";
    let description = [{
        Floating-point type that store in IEEE-754-2008 binary16 format.
    }];
}

def AtemIRFloat32Type
    : AtemIRFloatingPointType<"Float32", "float32"> {
    let summary = "AtemIR float32 type";
    let description = [{
        Floating-point type that store in IEEE-754-2008 binary32 format.
    }];
}

def AtemIRFloat64Type
    : AtemIRFloatingPointType<"Float64", "float64"> {
    let summary = "AtemIR float64 type";
    let description = [{
        Floating-point type that store in IEEE-754-2008 binary64 format.
    }];
}

def AtemIRFloat80Type
    : AtemIRFloatingPointType<"Float80", "float80"> {
    let summary = "AtemIR float80 type";
    let description = [{
        Floating-point type that store in x87 80-bit floating-point format.
    }];
}

def AtemIRFloat128Type
    : AtemIRFloatingPointType<"Float128", "float128"> {
    let summary = "AtemIR float128 type";
    let description = [{
        Floating-point type that store in IEEE-754-2008 binary128 format.
    }];
}

def AtemIRAnyType : AnyTypeOf<[
    AtemIRIntegerType, AtemIRBooleanType
]>;

#endif //ATEM_IR_TYPES_TD