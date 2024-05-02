#ifndef ATEM_IR_TYPES_TD
#define ATEM_IR_TYPES_TD

include "AtemIR/Dialect/include/AtemIRDialect.td"

include "AtemIR/Interfaces/include/AtemIRFPTypeInterface.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

class AtemIRType<string name, string typeMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<AtemIRDialect, name, traits, baseCppClass> {
    let mnemonic = typeMnemonic;
}

def IntegerType : 
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

def BooleanType :
    AtemIRType<"Boolean", "bool", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

    let summary = "AtemIR bool type";
    let description = [{
        `atemir.bool` represents Atem bool type.
    }];

    let hasCustomAssemblyFormat = 1;
}

class FloatingPointType<string name, string mnemonic>
    : AtemIRType<name, mnemonic, 
    [
        DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
        DeclareTypeInterfaceMethods<AtemIRFPTypeInterface>,
    ]> {}

def Float16Type
    : FloatingPointType<"Float16", "float16"> {
    let summary = "AtemIR float16 type";
    let description = [{
        Floating-point type that store in IEEE-754-2008 binary16 format.
    }];
}

def Float32Type
    : FloatingPointType<"Float32", "float32"> {
    let summary = "AtemIR float32 type";
    let description = [{
        Floating-point type that store in IEEE-754-2008 binary32 format.
    }];
}

def Float64Type
    : FloatingPointType<"Float64", "float64"> {
    let summary = "AtemIR float64 type";
    let description = [{
        Floating-point type that store in IEEE-754-2008 binary64 format.
    }];
}

def Float80Type
    : FloatingPointType<"Float80", "float80"> {
    let summary = "AtemIR float80 type";
    let description = [{
        Floating-point type that store in x87 80-bit floating-point format.
    }];
}

def Float128Type
    : FloatingPointType<"Float128", "float128"> {
    let summary = "AtemIR float128 type";
    let description = [{
        Floating-point type that store in IEEE-754-2008 binary128 format.
    }];
}

def AtemIRAnyType : AnyTypeOf<[
    IntegerType, BooleanType, Float16Type, Float32Type, Float64Type, Float80Type,
    Float128Type
]>;

#endif //ATEM_IR_TYPES_TD