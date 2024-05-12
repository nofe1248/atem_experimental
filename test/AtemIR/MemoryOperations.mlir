module {
    atemir.function @main() -> !atemir.unit {
        %0 = atemir.allocate !atemir.int<s, 64> -> !atemir.ptr<!atemir.int<s, 64>>, ["count", init] {alignment = 4}
        %1 = atemir.load %0 : !atemir.ptr<!atemir.int<s, 64>> -> !atemir.int<s, 64>
        %2 = atemir.constant #atemir.int<42> : !atemir.int<s, 64>
        atemir.store %2 : !atemir.int<s, 64>, %0 : !atemir.ptr<!atemir.int<s, 64>>
        atemir.return
    }
}