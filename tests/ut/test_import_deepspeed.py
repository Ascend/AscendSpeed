import unittest
from wrapt_timeout_decorator import timeout


class ImportTestCase(unittest.TestCase):
    
    @timeout(1200)
    def test_import(self):
        import deepspeed

