import fileinput
import re

verbose = False

def is_number(x):
  try:
    int(x)
    return True
  except:
    return False

def cleanup_fancy_op_names(op):
  op = op.split('(')[1]
  op = op[0] + re.sub("\_[0-9a-zA-Z]+","", op[1:])
  op = re.sub("\/","",op)
  return op

def extract_op_from_line(line):
  has_timestamp = is_number(line[0:4])
  if has_timestamp:
    op = line.split(']')[1].split(')')[0]
  else:
    op = line.split(')')[0]
  return cleanup_fancy_op_names(op)

def is_two_words_op(op):
  if " " in op:
    return True
  return False

def is_device_placement_line(line):
  wrong_line_keywords = [ "ignoring", "failed", "mismatch" , "error" ]
  if "job:localhost" in line:
    for k in wrong_line_keywords:
      if k in line.lower():
        return False
    return True

cpu_ops = set()
ngraph_ops = set()

number = 1

for line in fileinput.input():
  try:
    if is_device_placement_line(line):  
      device = line.split(':')[-2]
      op = extract_op_from_line(line)
      if is_two_words_op(op):
        continue 
      if device == "CPU":
        cpu_ops.add(op)
      else:
        ngraph_ops.add(op)
  except Exception as e:
    if (verbose):
      print (e) 
      print("Parsing error in line: ", number,":", line) 
  number = number + 1

only_ngraph = ngraph_ops - cpu_ops
only_cpu = cpu_ops - ngraph_ops
common_ops = ngraph_ops.intersection(cpu_ops)

print ("Found {} kind of ops on CPU".format(len(cpu_ops)))
print ("Found {} kind of ops on NGRAPH".format(len(ngraph_ops)))
print ("Found {} kind of ops on both devices".format(len(common_ops)))
print ("CPU exclusive ops {}".format(len(only_cpu)))
print ("NGRAPH exclusive ops {} ".format(len(only_ngraph)))


# match extraced ops set with standard ops list
with open("ops_list.txt") as f:
    ops_list = f.readlines()
ops_list = [x.strip() for x in ops_list] 

print ("Loaded {} standard ops".format(len(ops_list))) 


def check(op, ops_list):
  if op in ops_list:
    return ("X")
  else:
    return("")

print ("CSV with device placement of standard ops:")
print ("op name, NGRAPH, CPU")

for op in ops_list:
  print (op + "," + check(op, ngraph_ops) + "," + check(op, cpu_ops))
