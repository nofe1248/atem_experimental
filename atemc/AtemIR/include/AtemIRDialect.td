#ifndef ATEM_IR_DIALECT_TD
#define ATEM_IR_DIALECT_TD

include "mlir/IR/OpBase.td"

def AtemIRDialect : Dialect {
    let name = "AtemIR";
    let cppNamespace = "::atemir";
    let summary = "A high-level dialect which is suitable for optimization and analytics.";
    let description = [{
        The Atem language is a expression-based, multi-paradigm, general-purpose language.
        This dialect provides a high-level intermediate representation of the language by
        resembling the AST.
    }];
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

#endif  //ATEM_IR_DIALECT_TD