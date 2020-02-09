#!/usr/bin/env python
import os
import sys

def triangulate_obj(in_path, out_path):
    with open(os.path.expanduser(in_path)) as f:
        lines = []
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                lines.append(line)
                continue
            if tokens[0] == 'f':
                if len(tokens) == 4:
                    lines.append(line)
                else:
                    for i in range(len(tokens)-3):
                        lines.append('f %s %s %s\n'%(
                                tokens[1], tokens[i+2], tokens[i+3]))
            else:
                lines.append(line)
    
    out = ''.join(lines)
    with open(os.path.expanduser(out_path), 'w') as f:
        f.write(out)

if __name__ == '__main__':
    triangulate_obj(sys.argv[1], sys.argv[2])
