#ifndef ATEM_IR_ATTRS_TD
#define ATEM_IR_ATTRS_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "AtemIR/include/AtemIRDialect.td"

class AtemIRAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AtemIRDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif  //ATEM_IR_ATTRS_TD