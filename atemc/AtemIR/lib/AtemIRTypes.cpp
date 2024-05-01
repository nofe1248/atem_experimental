#include "AtemIR/include/AtemIRDialect.h"
#include "AtemIR/include/AtemIRTypes.h"

using namespace atemir;

void AtemIRDialect::registerTypes() {
    // Register tablegen'd types.
    addTypes<
        #define GET_TYPEDEF_LIST
        #include "AtemIRTypes.cpp.inc"
    >();
}