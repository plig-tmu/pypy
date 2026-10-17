from rpython.jit.metainterp.test.test_callee_frame import CalleeFrameTests
from rpython.jit.backend.x86.test.test_basic import Jit386Mixin

class TestCalleeFrame(Jit386Mixin, CalleeFrameTests):
    # for the individual tests see
    # ====> ../../../metainterp/test/test_callee_frame.py
    pass
