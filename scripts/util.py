import subprocess
import hashlib

# execute a command and return the output
def run_cmd(cmd):
   cmd_hash = hashlib.md5(cmd.encode()).hexdigest()[0:10]
   file_name = "/tmp/{cmd_hash}_cmd.txt".format(cmd_hash=cmd_hash)
   # subprocess.call("rm -f /tmp/*_cmd.txt", shell=True)  
   cmd = cmd + " | tee {file_name}".format(file_name=file_name)
   print("Running command: " + cmd)
   subprocess.call(cmd, shell=True) 
   with open(file_name, "r") as f:
    return f.read()