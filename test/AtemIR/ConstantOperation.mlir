module {
    atemir.function @main() -> i32 {
        %0 = atemir.constant #atemir.int<12>: !atemir.int<s, 64>
        %1 = atemir.constant #atemir.bool<true>: !atemir.bool
        %2 = atemir.constant #atemir.fp<1.14514>: !atemir.float64
        atemir.return %0: !atemir.int<s, 64>
    }
}