#ifndef ATEM_IR_DIALECT_TD
#define ATEM_IR_DIALECT_TD

include "mlir/IR/OpBase.td"

def AtemIRDialect : Dialect {
    let name = "AtemIR";
    let cppNamespace = "::AtemIR";
    let summary = "Atem IR Dialect";
}

#endif