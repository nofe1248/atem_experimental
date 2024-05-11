#ifndef ATEM_IR_ATTRS_TD
#define ATEM_IR_ATTRS_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

include "AtemIR/Dialect/include/AtemIRDialect.td"

class AtemIRAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AtemIRDialect, name, traits> {
    let mnemonic = attrMnemonic;
}

def AtemUnitAttr
    : AtemIRAttr<"Unit", "unit", [TypedAttrInterface]> {
    let summary = "Represent a unit value";
    let description = [{
        The `#atemir.unit` represents a unit value.
    }];

    let parameters = (ins AttributeSelfTypeParameter<"", "::atemir::UnitType">:$type);

    let assemblyFormat = [{}];
}

def BooleanAttr
    : AtemIRAttr<"Boolean", "bool", [TypedAttrInterface]> {
    let summary = "Represent true/false for !atemir.bool types";
    let description = [{
        The BooleanAttr represents a 'true' or 'false' value.
    }];

    let parameters = (ins AttributeSelfTypeParameter<"", "::atemir::BooleanType">:$type,"bool":$value);

    let assemblyFormat = [{
        `<` $value `>`
    }];
}

def IntegerAttr
    : AtemIRAttr<"Integer", "int", [TypedAttrInterface]> {
    let summary = "An attribute containing a integer value";
    let description = [{
        An integer attribute is a literal attribute that represents an integral
        value of the specified integer type.
    }];

    let parameters = (ins AttributeSelfTypeParameter<"">:$type, "mlir::APInt":$value);

    let builders = [
        AttrBuilderWithInferredContext<(ins "::mlir::Type":$type,
                                        "const ::mlir::APInt &":$value), [{
            return $_get(type.getContext(), type, value);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;

    let skipDefaultBuilders = 1;
}

def FloatingPointAttr
    : AtemIRAttr<"FloatingPoint", "fp", [TypedAttrInterface]> {
    let summary = "An attribute containing a floating-point value";
    let description = [{
        An fp attribute is a literal attribute that represents a floating-poing
        value of the specified floating-point type.
    }];

    let parameters = (ins AttributeSelfTypeParameter<"">:$type, "mlir::APFloat":$value);

    let builders = [
        AttrBuilderWithInferredContext<(ins "::mlir::Type":$type,
                                            "const ::mlir::APFloat &":$value), [{
        return $_get(type.getContext(), type, value);
        }]>,
    ];

    let extraClassDeclaration = [{
        static auto getZero(mlir::Type type) -> FloatingPointAttr;
    }];

    let assemblyFormat = [{
        `<` custom<FloatLiteral>($value, ref($type)) `>`
    }];
}

#endif  //ATEM_IR_ATTRS_TD