#ifndef ATEM_IR_OP_ENUMS_TD
#define ATEM_IR_OP_ENUMS_TD

include "mlir/IR/EnumAttr.td"

def UnaryOpKind_Neg : I32EnumAttrCase<"Neg", 1, "neg">;
def UnaryOpKind_Not : I32EnumAttrCase<"Not", 2, "not">;

def UnaryOpKind : I32EnumAttr<
    "UnaryOpKind",
    "Unary operation kind",
    [
        UnaryOpKind_Neg,
        UnaryOpKind_Not
    ]
> {
    let cppNamespace = "::atemir";
}

def BinaryOpKind_Mul : I32EnumAttrCase<"Mul", 1, "mul">;
def BinaryOpKind_Div : I32EnumAttrCase<"Div", 2, "div">;
def BinaryOpKind_Rem : I32EnumAttrCase<"Rem", 3, "rem">;
def BinaryOpKind_Add : I32EnumAttrCase<"Add", 4, "add">;
def BinaryOpKind_Sub : I32EnumAttrCase<"Sub", 5, "sub">;
def BinaryOpKind_And : I32EnumAttrCase<"And", 6, "and">;
def BinaryOpKind_Or : I32EnumAttrCase<"Or", 7, "or">;
def BinaryOpKind_Xor : I32EnumAttrCase<"Xor", 8, "xor">;

def BinaryOpKind : I32EnumAttr<
    "BinaryOpKind",
    "Binary operation kind",
    [
        BinaryOpKind_Mul,
        BinaryOpKind_Div,
        BinaryOpKind_Rem,
        BinaryOpKind_Add,
        BinaryOpKind_Sub,
        BinaryOpKind_And,
        BinaryOpKind_Or,
        BinaryOpKind_Xor
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