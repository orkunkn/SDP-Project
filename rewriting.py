rewriting_list = []

def dump_rewriting_list(filename, entries):
   with open(filename, 'w') as f:
     for entity, source, target in entries: 
       f.write(f"{entity},{source},{target}\n")

