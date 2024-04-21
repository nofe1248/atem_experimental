#ifndef ATEM_IR_OPS_TD
#define ATEM_IR_OPS_TD

include "mlir/IR/OpBase.td"
include "AtemIR/include/AtemIRDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class AtemIROp<string mnemonic, list<Trait> traits = []> :
  Op<AtemIRDialect, mnemonic, traits>;

#endif