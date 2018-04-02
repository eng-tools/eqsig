import subprocess
import py

about = {}
with open("eqsig/__about__.py") as fp:
    exec(fp.read(), about)

version = about['__version__']

failures = py.test.cmdline.main()
if failures == 0:
    subprocess.check_call(["git", "tag", version, "-m", "version %s" % version])
    subprocess.check_call(["git", "push", "--tags", "origin", "master"])