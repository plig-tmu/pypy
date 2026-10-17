"""Acceptance tests for the experimental 'callee_frame' JIT parameter.

Baseline tests pass today; the xfail(strict=True) ones describe the
targets and start passing (=> XPASS => failure) once implemented, at
which point remove the marker.

    python2 pytest.py rpython/jit/metainterp/test/test_callee_frame.py
"""
import py
from rpython.rlib.jit import JitDriver, hint, dont_look_inside
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp.resoperation import rop
from rpython.jit.metainterp.warmspot import get_stats

fields_passed = py.test.mark.xfail(strict=True,
    reason="callee_frame: fields not passed to call_assembler yet")
frame_virtual = py.test.mark.xfail(strict=True,
    reason="callee_frame: Frame not kept virtual in the callee yet")


# ---------- the toy interpreter: fib(n) with one virtualizable Frame ----------

class Frame(object):
    _virtualizable_ = ['n', 'acc']

    def __init__(self, n):
        self = hint(self, access_directly=True, fresh_virtualizable=True)
        self.n = n
        self.acc = 0

driver = JitDriver(greens=['code'], reds=['frame'], virtualizables=['frame'])

class Seen(object):
    log = []
seen = Seen()

@dont_look_inside
def peek(frame):
    # residual call that receives the frame => forces it via vable_token
    seen.log.append(frame.n * 100 + frame.acc)

class Boom(Exception):
    pass

def portal(code, frame):
    # code: 0 = plain fib, 1 = also peek() at n == 3,
    #       2 = also raise Boom at n == 0 (caught one level up),
    #       3 = rare branch at n == 3 whose frame state is used later
    driver.jit_merge_point(code=code, frame=frame)
    n = frame.n
    if n < 2:
        if code == 2 and n == 0:
            raise Boom
        return n
    if code == 1 and n == 3:
        frame.acc = 7
        peek(frame)
    if code == 3 and n == 3:
        frame.acc = 5
    try:
        a = portal(code, Frame(n - 1))
    except Boom:
        a = 0
    try:
        b = portal(code, Frame(n - 2))
    except Boom:
        b = 0
    return a + b + frame.acc

def main(code, n):
    del seen.log[:]
    return portal(code, Frame(n))


class CalleeFrameTests:

    def run(self, code, n, callee_frame):
        return self.meta_interp(main, [code, n], inline=True,
                                max_unroll_recursion=1,
                                callee_frame=callee_frame)

    def count_call_assembler(self):
        return sum(1 for loop in get_stats().get_all_loops()
                     for op in loop.operations
                     if rop.is_call_assembler(op.getopnum()))

    def frames_allocated_for_call_assembler(self):
        """Number of call_assembler ops that get an argument produced by a
        new_with_vtable in the same trace (= caller-side Frame allocation)."""
        count = 0
        for loop in get_stats().get_all_loops():
            news = set()
            for op in loop.operations:
                if op.getopnum() == rop.NEW_WITH_VTABLE:
                    news.add(op)
                elif rop.is_call_assembler(op.getopnum()):
                    if [arg for arg in op.getarglist() if arg in news]:
                        count += 1
        return count

    def frame_allocs_except_escapes(self):
        """Number of new_with_vtable(Frame) in the compiled loops (bridges
        are not recorded by get_stats()) whose result is not passed to a
        residual call, i.e. allocations that are not an escape point."""
        count = 0
        for loop in get_stats().get_all_loops():
            frames = [op for op in loop.operations
                      if op.getopnum() == rop.NEW_WITH_VTABLE and
                         'Frame' in op.getdescr().S._name]
            escaped = set()
            for op in loop.operations:
                if (rop.is_call(op.getopnum()) and
                        not rop.is_call_assembler(op.getopnum())):
                    escaped.update(op.getarglist())
            count += len([op for op in frames if op not in escaped])
        return count

    # ---- baseline: pass today, must keep passing ----

    def test_toy_uses_call_assembler(self):
        assert self.run(0, 15, 0) == main(0, 15)
        assert self.count_call_assembler() > 0
        assert self.frames_allocated_for_call_assembler() > 0

    def test_results_flag_off(self):
        for code in [0, 1, 2, 3]:
            assert self.run(code, 15, 0) == main(code, 15)

    def test_results_flag_on(self):
        for code in [0, 1, 2, 3]:
            assert self.run(code, 15, 1) == main(code, 15)

    def test_force_sees_fields_flag_on(self):
        expected = (main(1, 15), seen.log[:])
        assert self.run(1, 15, 1) == expected[0]
        assert seen.log == expected[1]

    def test_caller_reads_callee_frame_after_call(self):
        # like test_recursive.test_directly_call_assembler_virtualizable:
        # the caller reads the callee frame after call_assembler returns,
        # so callee_frame must fall back to the materialized Frame here.
        def portal2(code, frame):
            driver.jit_merge_point(code=code, frame=frame)
            n = frame.n
            if n < 2:
                frame.acc = n
                return n
            sub = Frame(n - 1)
            a = portal2(code, sub)
            frame.acc = sub.acc + a
            return a + portal2(code, Frame(n - 2))
        def main2(code, n):
            f = Frame(n)
            return portal2(code, f) * 1000 + f.acc
        res = self.meta_interp(main2, [0, 12], inline=True,
                               max_unroll_recursion=1, callee_frame=1)
        assert res == main2(0, 12)
        assert self.count_call_assembler() > 0

    # ---- target: no caller-side Frame allocation for call_assembler ----

    @fields_passed
    def test_no_frame_alloc_before_call_assembler(self):
        assert self.run(0, 15, 1) == main(0, 15)
        assert self.count_call_assembler() > 0
        assert self.frames_allocated_for_call_assembler() == 0

    # ---- target: Frame stays virtual in the callee; slow paths correct ----

    @frame_virtual
    def test_frame_virtual_in_callee(self):
        assert self.run(0, 15, 1) == main(0, 15)
        assert self.count_call_assembler() > 0
        assert self.frame_allocs_except_escapes() == 0

    @frame_virtual
    def test_guard_failure_in_callee(self):
        # code 3 takes a rare branch at n == 3 and later reads frame.acc:
        # the guard failure must rebuild the Frame with the right fields
        assert self.run(3, 15, 1) == main(3, 15)
        assert self.frame_allocs_except_escapes() == 0

    @frame_virtual
    def test_force_materializes_frame(self):
        expected = (main(1, 15), seen.log[:])
        assert expected[1]      # peek() did run
        assert self.run(1, 15, 1) == expected[0]
        assert seen.log == expected[1]
        assert self.frame_allocs_except_escapes() == 0

    @frame_virtual
    def test_exception_in_callee(self):
        assert self.run(2, 15, 1) == main(2, 15)
        assert self.frame_allocs_except_escapes() == 0


class TestLLtype(CalleeFrameTests, LLJitMixin):
    pass
