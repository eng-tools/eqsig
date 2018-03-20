import subprocess

about = {}
with open("eqsig/__about__.py") as fp:
    exec(fp.read(), about)

version = about['__version__']

subprocess.check_call(["git", "tag", version, "-m", "version %s" % version])
subprocess.check_call(["git", "push", "--tags", "origin", "master"])