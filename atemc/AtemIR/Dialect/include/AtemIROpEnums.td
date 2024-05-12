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

def MemoryOrderKind_Relaxed : I32EnumAttrCase<"Relaxed", 0, "relaxed">;
def MemoryOrderKind_Consume : I32EnumAttrCase<"Consume", 1, "consume">;
def MemoryOrderKind_Acquire : I32EnumAttrCase<"Acquire", 2, "acquire">;
def MemoryOrderKind_Release : I32EnumAttrCase<"Release", 3, "release">;
def MemoryOrderKind_AcqRel : I32EnumAttrCase<"AcquireRelease", 4, "acq_rel">;
def MemoryOrderKind_SeqCst : I32EnumAttrCase<"SequentiallyConsistent", 5, "seq_cst">;

def MemoryOrderKind : I32EnumAttr<
    "MemoryOrderKind",
    "Memory order",
    [
        MemoryOrderKind_Relaxed, 
        MemoryOrderKind_Consume, 
        MemoryOrderKind_Acquire,
        MemoryOrderKind_Release, 
        MemoryOrderKind_AcqRel, 
        MemoryOrderKind_SeqCst
    ]> {
  let cppNamespace = "::atemir";
}

// Externally visible function
def Global_ExternalLinkage : I32EnumAttrCase<"ExternalLinkage", 0, "external">;
// Available for inspection, not emission.
def Global_AvailableExternallyLinkage : I32EnumAttrCase<"AvailableExternallyLinkage", 1, "available_externally">;
// Keep one copy of function when linking (inline)
def Global_LinkOnceAnyLinkage : I32EnumAttrCase<"LinkOnceAnyLinkage", 2, "linkonce">;
// Same, but only replaced by something equivalent.
def Global_LinkOnceODRLinkage : I32EnumAttrCase<"LinkOnceODRLinkage", 3, "linkonce_odr">;
// Keep one copy of named function when linking (weak)
def Global_WeakAnyLinkage : I32EnumAttrCase<"WeakAnyLinkage", 4, "weak">;
// Same, but only replaced by something equivalent.
def Global_WeakODRLinkage : I32EnumAttrCase<"WeakODRLinkage", 5, "weak_odr">;
// Special purpose, only applies to global arrays
def Global_AppendingLinkage : I32EnumAttrCase<"AppendingLinkage", 6, "appending">;
// Rename collisions when linking (static functions).
def Global_InternalLinkage : I32EnumAttrCase<"InternalLinkage", 7, "internal">;
// Like Internal, but omit from symbol table, prefix it with
// "atemir_" to prevent clash with MLIR's symbol "private".
def Global_PrivateLinkage : I32EnumAttrCase<"PrivateLinkage", 8, "atemir_private">;
// ExternalWeak linkage description.
def Global_ExternalWeakLinkage : I32EnumAttrCase<"ExternalWeakLinkage", 9, "extern_weak">;
// Tentative definitions.
def Global_CommonLinkage : I32EnumAttrCase<"CommonLinkage", 10, "common">;

/// An enumeration for the kinds of linkage for global values.
def GlobalLinkageKind : I32EnumAttr<
    "GlobalLinkageKind",
    "Linkage type",
    [
        Global_ExternalLinkage, 
        Global_AvailableExternallyLinkage,
        Global_LinkOnceAnyLinkage, 
        Global_LinkOnceODRLinkage,
        Global_WeakAnyLinkage, 
        Global_WeakODRLinkage,
        Global_AppendingLinkage,
        Global_InternalLinkage, 
        Global_PrivateLinkage,
        Global_ExternalWeakLinkage, 
        Global_CommonLinkage
     ]> {
    let cppNamespace = "::atemir";
}

def TLS_GeneralDynamic : I32EnumAttrCase<"GeneralDynamic", 0, "tls_dyn">;
def TLS_LocalDynamic : I32EnumAttrCase<"LocalDynamic", 1, "tls_local_dyn">;
def TLS_InitialExec : I32EnumAttrCase<"InitialExec", 2, "tls_init_exec">;
def TLS_LocalExec : I32EnumAttrCase<"LocalExec", 3, "tls_local_exec">;

def TLSModel : I32EnumAttr<
    "TLS_Model",
    "TLS model",
    [
        TLS_GeneralDynamic, 
        TLS_LocalDynamic, 
        TLS_InitialExec, 
        TLS_LocalExec
    ]> {
    let cppNamespace = "::atemir";
}

#endif //ATEM_IR_OP_ENUMS_TD