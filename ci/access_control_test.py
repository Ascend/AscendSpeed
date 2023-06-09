import os
import sys
from pathlib import Path

#=============================
# ST test, run with shell
#=============================
def success_check(res):
    if res != 0:
        sys.exit(1)

class ST_Test:
    def __init__(self):
        BASE_DIR = Path(__file__).absolute().parent.parent
        TEST_DIR = os.path.join(BASE_DIR, 'tests')

        gpt_shell_file = os.path.join(TEST_DIR, "st", "test_gpt", "test_gpt_ptd.sh")
        llama_shell_file = os.path.join(TEST_DIR, "st", "test_llama", "test_llama_ptd.sh")

        self.shell_file_list = [gpt_shell_file, llama_shell_file]

    def run_shell(self):
        for shell_file in self.shell_file_list:
            success_check(os.system("sh {}".format(shell_file)))

#===============================================
# UT test, run with pytest, waiting for more ...
#===============================================

if __name__ == "__main__":
    st_test = ST_Test()
    st_test.run_shell()
