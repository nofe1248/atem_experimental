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

def UnitType :
    AtemIRType<"Unit", "unit"> {
    let summary = "AtemIR unit type";
    let description = [{
        `!atemir.unit` type represents a unit type in Atem.
    }];
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

class UInt<int width>
    : Type<
        And<[
            CPred<"$_self.isa<::atemir::IntegerType>()">,
            CPred<"$_self.cast<::atemir::IntegerType>().isUnsigned()">,
            CPred<"$_self.cast<::atemir::IntegerType>().getWidth() == " # width>
        ]>
    > {
    int bitwidth = width;
}

class SInt<int width>
    : Type<
        And<[
            CPred<"$_self.isa<::atemir::IntegerType>()">,
            CPred<"$_self.cast<::atemir::IntegerType>().isSigned()">,
            CPred<"$_self.cast<::atemir::IntegerType>().getWidth() == " # width>
        ]>
    > {
    int bitwidth = width;
}

def UInt1 : UInt<1>;
def UInt8 : UInt<8>;
def UInt16 : UInt<16>;
def UInt31 : UInt<32>;
def UInt64 : UInt<64>;
def UInt128 : UInt<128>;

def SInt1 : SInt<1>;
def SInt8 : SInt<8>;
def SInt16 : SInt<16>;
def SInt31 : SInt<32>;
def SInt64 : SInt<64>;
def SInt128 : SInt<128>;

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

def AtemFunctionType
    : AtemIRType<"Function", "function"> {
    let summary = "AtemIR function type";
    let description = [{
        `!atemir.function` is a function type. It contains a list of return type and a
        list of parameter types.

        Example:
        ```mlir
        ```
    }];

    let parameters = (
        ins ArrayRefParameter<"Type">:$inputs,
        ArrayRefParameter<"Type">:$results
    );

    let assemblyFormat = [{
        `<` `(` $inputs `)` `->` $results `>`
    }];

    let builders = [
        TypeBuilderWithInferredContext<
        (ins "::mlir::ArrayRef<::mlir::Type>":$inputs, "::mlir::ArrayRef<::mlir::Type>":$results), [{
            return $_get(results[0].getContext(), inputs, results);
        }]>
    ];

    let extraClassDeclaration = [{
        /// Returns the `i`th input operand type. Asserts if out of bounds.
        auto getInput(unsigned i) const -> ::mlir::Type { return getInputs()[i]; }

        /// Returns the number of arguments to the function.
        auto getNumInputs() const -> unsigned { return getInputs().size(); }

        /// Returns whether the function is returns void.
        auto isReturningUnit() const -> bool;

        /// Returns a clone of this function type with the given argument
        /// and result types.
        auto clone(::mlir::TypeRange inputs, ::mlir::TypeRange results) const
         -> ::atemir::FunctionType;
    }];
}

def PointerType
    : AtemIRType<"Pointer", "ptr", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
    let summary = "AtemIR pointer type";
    let description = [{
        `atemir.ptr` is a raw pointer in Atem.
    }];

    let parameters = (ins "::mlir::Type":$pointee);

    let assemblyFormat = [{
        `<` $pointee `>`
    }];
}

def AtemIRAnyFloatingPointType : AnyTypeOf<[
    Float16Type, Float32Type, Float64Type, Float80Type, Float128Type
]>;

def AtemIRAnyType : AnyTypeOf<[
    IntegerType, BooleanType, Float16Type, Float32Type, Float64Type, Float80Type,
    Float128Type, AtemFunctionType, UnitType, PointerType
]>;

#endif //ATEM_IR_TYPES_TD