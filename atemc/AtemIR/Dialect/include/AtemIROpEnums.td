#ifndef ATEM_IR_OP_ENUMS_TD
#define ATEM_IR_OP_ENUMS_TD

include "mlir/IR/EnumAttr.td"

def UnaOpKind_Neg : I32EnumAttrCase<"Neg", 1, "neg">;

def UnaryOpKind : I32EnumAttr<
    "UnaryOpKind",
    "Unary operation kind",
    [
        UnaOpKind_Neg
    ]
> {
    let cppNamespace = "::atemir";
}

def BinOpKind_Mul : I32EnumAttrCase<"Mul", 1, "mul">;
def BinOpKind_Div : I32EnumAttrCase<"Div", 2, "div">;
def BinOpKind_Rem : I32EnumAttrCase<"Rem", 3, "rem">;
def BinOpKind_Add : I32EnumAttrCase<"Add", 4, "add">;
def BinOpKind_Sub : I32EnumAttrCase<"Sub", 5, "sub">;

def BinaryOpKind : I32EnumAttr<
    "BinaryOpKind",
    "Binary operation kind",
    [
        BinOpKind_Mul,
        BinOpKind_Div,
        BinOpKind_Rem,
        BinOpKind_Add,
        BinOpKind_Sub
    ]
> {
    let cppNamespace = "::atemir";
}

def CompareOpKind_LT : I32EnumAttrCase<"lt", 1>;
def CompareOpKind_LE : I32EnumAttrCase<"le", 2>;
def CompareOpKind_GT : I32EnumAttrCase<"gt", 3>;
def CompareOpKind_GE : I32EnumAttrCase<"ge", 4>;
def CompareOpKind_EQ : I32EnumAttrCase<"eq", 5>;
def CompareOpKind_NE : I32EnumAttrCase<"ne", 6>;

def CompareOpKind : I32EnumAttr<
    "CompareOpKind",
    "compare operation kind",
    [
        CompareOpKind_LT, 
        CompareOpKind_LE, 
        CompareOpKind_GT,
        CompareOpKind_GE, 
        CompareOpKind_EQ, 
        CompareOpKind_NE]
> {
    let cppNamespace = "::atemir";
}

#endif //ATEM_IR_OP_ENUMS_TD