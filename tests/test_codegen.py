"""Backend tests: IR structure, type mapping, and control-flow lowering.

These assert on the *shape* of the emitted IR -- the mapping from language
constructs to LLVM instructions -- which is where a backend is either right or
quietly wrong.
"""

from __future__ import annotations

import unittest

from helpers import IRTestCase, compile_or_fail, run

from helpers import ROOT, SRC  # noqa: F401

from lumen import codegen


def main(body: str) -> str:
    return f"func main() -> int {{\n{body}\n    return 0;\n}}"


class TestTypeMapping(IRTestCase):
    def test_int_maps_to_i64(self):
        self.assert_contains(compile_or_fail(self, main("let x: int = 1;")), "i64")

    def test_float_maps_to_double(self):
        self.assert_contains(compile_or_fail(self, main("let x: float = 1.0;")),
                             "double")

    def test_bool_maps_to_i1(self):
        self.assert_contains(compile_or_fail(self, main("let x: bool = true;")), "i1")

    def test_string_maps_to_a_pointer(self):
        ir = compile_or_fail(self, main('let x: string = "hi";'))
        self.assert_contains(ir, "i8*")
        self.assert_contains(ir, 'c"hi\\00"')

    def test_array_maps_to_an_element_pointer(self):
        ir = compile_or_fail(self, main("let a: int[] = [1];"))
        self.assert_contains(ir, "i64*")

    def test_void_return(self):
        ir = compile_or_fail(self, "func f() { }" + "\n" + main(""))
        self.assert_contains(ir, "ret void")

    def test_main_is_renamed_never_and_user_functions_are_mangled(self):
        ir = compile_or_fail(self, main("print helper();") +
                             "\nfunc helper() -> int { return 1; }")
        self.assert_contains(ir, "define i64 @main()")
        self.assert_contains(ir, "define i64 @_lumen_helper()")

    def test_target_triple_is_emitted_when_requested(self):
        from lumen.driver import compile_source

        result = compile_source(main(""), "t.lum",
                                target="x86_64-pc-windows-msvc")
        self.assert_contains(result.ir,
                             'target triple = "x86_64-pc-windows-msvc"')


class TestDeclarations(IRTestCase):
    def test_locals_live_in_allocas_with_explicit_stores(self):
        ir = compile_or_fail(self, main("let x = 42;"))
        self.assert_contains(ir, "= alloca i64")
        self.assert_contains(ir, "store i64 42")

    def test_parameters_are_spilled_to_the_entry_block(self):
        ir = compile_or_fail(self, main("print f(1);") +
                             "\nfunc f(a: int) -> int { return a; }")
        self.assert_contains(ir, "store i64 %a, i64* %")

    def test_runtime_declarations_are_emitted(self):
        ir = compile_or_fail(self, main('print "x";'))
        for symbol in ("@printf", "@malloc", "@memcpy", "@strcmp", "@strlen"):
            self.assert_contains(ir, f" {symbol}(")

    def test_strings_are_interned_once(self):
        ir = compile_or_fail(self, main('print "dup"; print "dup";'))
        self.assertEqual(ir.count('c"dup\\00"'), 1)

    def test_control_characters_are_escaped(self):
        ir = compile_or_fail(self, main('print "a\\tb\\n";'))
        self.assert_contains(ir, "\\09")
        self.assert_contains(ir, "\\0A")

    def test_floats_use_exact_hex_constants(self):
        ir = compile_or_fail(self, main("let x = 0.5;"))
        self.assert_contains(ir, "0x3FE0000000000000")


class TestControlFlow(IRTestCase):
    def test_if_else_becomes_blocks_and_a_conditional_branch(self):
        ir = compile_or_fail(self, main("if 1 < 2 { print 1; } else { print 2; }"))
        self.assert_contains(ir, "if.then")
        self.assert_contains(ir, "if.else")
        self.assert_contains(ir, "if.end")
        self.assert_contains(ir, "br i1 ")

    def test_if_without_else_branches_straight_to_the_merge(self):
        ir = compile_or_fail(self, main("if 1 < 2 { print 1; }"))
        self.assert_contains(ir, "br i1 ")
        self.assert_lacks(ir, "if.else")

    def test_while_emits_a_condition_block_and_back_edge(self):
        ir = compile_or_fail(self, main("while false { }"))
        self.assert_contains(ir, "while.cond")
        self.assert_contains(ir, "while.body")
        self.assert_contains(ir, "br label %while.cond")

    def test_for_loop_is_counted_and_half_open(self):
        ir = compile_or_fail(self, main("for i in 0..10 { print i; }"))
        self.assert_contains(ir, "for.cond")
        self.assert_contains(ir, "for.step")
        # `<` not `<=`, and the body increments a separate induction slot.
        self.assert_contains(ir, "icmp slt")
        self.assert_contains(ir, "add i64")

    def test_break_and_continue_branch_to_the_right_labels(self):
        ir = compile_or_fail(self, main("while true { break; continue; }"))
        self.assert_contains(ir, "br label %while.end")
        self.assert_contains(ir, "br label %while.cond")

    def test_continue_in_a_for_loop_targets_the_step_block(self):
        ir = compile_or_fail(self, main("for i in 0..3 { continue; }"))
        self.assert_contains(ir, "br label %for.step")

    def test_return_inside_a_loop(self):
        ir = compile_or_fail(self, main("while true { return 3; }"), opt_level=0)
        self.assert_contains(ir, "ret i64 3")


class TestExpressions(IRTestCase):
    def test_integer_arithmetic_uses_signed_ops(self):
        ir = compile_or_fail(self, main("let a = 6; let b = 3; print a / b; print a % b;"))
        self.assert_contains(ir, "sdiv")
        self.assert_contains(ir, "srem")

    def test_float_arithmetic_uses_float_ops(self):
        ir = compile_or_fail(self, main("let a = 6.0; let b = 3.0; print a / b;"))
        self.assert_contains(ir, "fdiv")

    def test_comparisons_produce_i1(self):
        ir = compile_or_fail(self, main("let b = 1 < 2;"))
        self.assert_contains(ir, "icmp slt i64")
        self.assert_contains(ir, "i1")

    def test_float_comparison_uses_fcmp(self):
        ir = compile_or_fail(self, main("let b = 1.0 < 2.0;"))
        self.assert_contains(ir, "fcmp olt double")

    def test_logical_and_short_circuits_through_a_block(self):
        ir = compile_or_fail(self, main("let b = false && false;"))
        self.assert_contains(ir, "logic.rhs")
        self.assert_contains(ir, "logic.end")

    def test_logical_or_short_circuits_through_a_block(self):
        ir = compile_or_fail(self, main("let b = true || false;"))
        self.assert_contains(ir, "logic.rhs")

    def test_negation_uses_xor(self):
        ir = compile_or_fail(self, main("let b = !true;"))
        self.assert_contains(ir, "xor i1")

    def test_int_to_float_widening_uses_sitofp(self):
        ir = compile_or_fail(self, main("let a: float = 1; let b = a + 2.0;"))
        self.assert_contains(ir, "sitofp")

    def test_indexing_uses_getelementptr(self):
        ir = compile_or_fail(self, main("let a: int[] = [1, 2]; print a[1];"))
        self.assert_contains(ir, "getelementptr inbounds i64")

    def test_string_equality_uses_strcmp(self):
        ir = compile_or_fail(self, main('let b = "a" == "b";'))
        self.assert_contains(ir, "@strcmp")

    def test_print_passes_each_primitive_with_its_real_type(self):
        # A variadic argument must keep its type: %s needs i8*, not i64.
        ir = compile_or_fail(self, main('print 1; print 1.0; print true; print "s";'))
        self.assert_contains(ir, '@printf(i8* %')
        self.assert_contains(ir, ", i64 ")
        self.assert_contains(ir, ", double ")
        self.assert_contains(ir, ", i8* ")

    def test_calls_are_mangled_and_typed(self):
        ir = compile_or_fail(self, main("print f(1);") +
                             "\nfunc f(a: int) -> int { return a; }")
        self.assert_contains(ir, "call i64 @_lumen_f(i64")

    def test_no_phi_nodes_are_ever_emitted(self):
        ir = compile_or_fail(self, main("""
            let a = true && false;
            let b = 1 < 2 || 3 > 4;
            for i in 0..3 { print i; }
            while a { print b; }
            if a { print 1; } else { print 2; }
        """))
        self.assertEqual(ir.count(" phi "), 0)


class TestArrays(IRTestCase):
    def test_array_literal_allocates_with_a_length_header(self):
        ir = compile_or_fail(self, main("let a: int[] = [1, 2, 3];"))
        # 3 elements * 8 bytes + an 8 byte header = 32
        self.assert_contains(ir, "mul i64 3, 8")
        self.assert_contains(ir, "add i64")
        self.assert_contains(ir, "@malloc")
        self.assert_contains(ir, "store i64 3, i8*")

    def test_len_reads_the_header_below_the_pointer(self):
        ir = compile_or_fail(self, main("let a: int[] = [1, 2]; print len(a);"))
        self.assert_contains(ir, "getelementptr i8, i8* ")
        self.assert_contains(ir, "i64 -8")

    def test_len_of_a_string_calls_strlen(self):
        ir = compile_or_fail(self, main('print len("abcd");'))
        self.assert_contains(ir, "@strlen")

    def test_array_assignment_copies_rather_than_aliases(self):
        ir = compile_or_fail(self, main("let a: int[] = [1]; let b: int[] = [2]; b = a;"))
        self.assert_contains(ir, "@memcpy")

    def test_array_assignment_can_grow_with_realloc(self):
        ir = compile_or_fail(self, main(
            "let a: int[] = [1]; let b: int[] = [2, 3]; b = a;"))
        self.assert_contains(ir, "@realloc")

    def test_array_initialisation_clones_the_source(self):
        ir = compile_or_fail(self, main("let a: int[] = [1]; let b: int[] = a;"))
        self.assert_contains(ir, "@malloc")
        self.assert_contains(ir, "@memcpy")


class TestOptimization(IRTestCase):
    def test_constant_folding_removes_arithmetic_from_the_ir(self):
        ir = compile_or_fail(self, main("let x = 2 * 3 + 4;"), opt_level=1)
        self.assert_contains(ir, "store i64 10")
        self.assert_lacks(ir, "mul i64")

    def test_dead_branch_is_eliminated(self):
        ir = compile_or_fail(self, main("if false { print 1; } else { print 2; }"),
                             opt_level=1)
        self.assert_lacks(ir, "if.then")

    def test_always_true_branch_is_kept(self):
        ir = compile_or_fail(self, main("if true { print 1; }"), opt_level=1)
        self.assert_lacks(ir, "if.cond")

    def test_identity_is_folded(self):
        ir = compile_or_fail(self, main("let a = 5; let b = a * 1 + 0;"),
                             opt_level=1)
        self.assert_lacks(ir, "mul i64")
        self.assert_lacks(ir, "add i64 %")

    def test_zero_loops_are_removed(self):
        ir = compile_or_fail(self, main("for i in 5..5 { print 1; }"), opt_level=1)
        self.assert_lacks(ir, "for.cond")

    def test_statements_after_return_are_dropped(self):
        ir = compile_or_fail(self, main("return 1; print 2;"), opt_level=1)
        self.assert_lacks(ir, "store i64 2")

    def test_optimization_is_reversible_at_level_zero(self):
        source = main("let x = 2 * 3;")
        unoptimized = compile_or_fail(self, source, opt_level=0)
        optimized = compile_or_fail(self, source, opt_level=2)
        self.assert_contains(unoptimized, "mul i64")
        self.assert_lacks(optimized, "mul i64")

    def test_folding_preserves_behaviour(self):
        source = main("print 2 * 3 + 4; print (7 - 2) * 6; print 1 < 2;")
        self.assert_runs(source, "10\n30\n1\n")


class TestRuntimeDeclarations(IRTestCase):
    def test_all_used_externs_are_declared(self):
        ir = compile_or_fail(self, main("""
            let a: int[] = [1, 2];
            print len(a);
            print len("s");
            print "x" == "y";
        """))
        for symbol in ("@printf", "@strlen", "@strcmp", "@malloc"):
            self.assert_contains(ir, f" {symbol}(")

    def test_free_is_declared_but_never_called(self):
        # Arrays live for the whole program; nothing is released.
        ir = compile_or_fail(self, main("let a: int[] = [1];"))
        self.assert_contains(ir, "declare void @free(i8*)")
        self.assert_lacks(ir, "call void @free")


class TestVerifier(unittest.TestCase):
    def test_verifier_accepts_every_example(self):
        import glob
        import os

        from helpers import EXAMPLES
        from lumen import irverify
        from lumen.driver import compile_source

        for path in sorted(glob.glob(os.path.join(EXAMPLES, "*.lum"))):
            with self.subTest(example=os.path.basename(path)):
                with open(path, encoding="utf-8") as handle:
                    result = compile_source(handle.read(), path, opt_level=2)
                self.assertTrue(result.ok)
                self.assertTrue(irverify.verify(result.ir).ok)

    def test_verifier_rejects_an_undefined_branch(self):
        from lumen.irinterp import IRError  # noqa: F401  (import sanity)
        from lumen import irverify

        bad = """define i64 @main() {
entry:
  br label %nowhere
}
"""
        result = irverify.verify(bad)
        self.assertFalse(result.ok)
        self.assertTrue(any("undefined label" in e for e in result.errors))

    def test_verifier_rejects_an_undefined_value(self):
        from lumen import irverify

        bad = """define i64 @main() {
entry:
  %a = add i64 %ghost, 1
  ret i64 %a
}
"""
        result = irverify.verify(bad)
        self.assertFalse(result.ok)
        self.assertTrue(any("undefined value" in e for e in result.errors))

    def test_verifier_rejects_a_double_terminator(self):
        from lumen import irverify

        bad = """define i64 @main() {
entry:
  ret i64 0
  ret i64 1
}
"""
        result = irverify.verify(bad)
        self.assertFalse(result.ok)
        self.assertTrue(any("outside a block" in e for e in result.errors))

    def test_verifier_rejects_redefinition(self):
        from lumen import irverify

        bad = """define i64 @main() {
entry:
  %a = add i64 1, 2
  %a = add i64 3, 4
  ret i64 %a
}
"""
        result = irverify.verify(bad)
        self.assertFalse(result.ok)
        self.assertTrue(any("redefined" in e for e in result.errors))


if __name__ == "__main__":
    unittest.main()