import sys
from termcolor import colored, cprint

with open(sys.argv[1]) as f1, open(sys.argv[2]) as f2:
    for a, b in zip(f1, f2):
        la, lb = a.rstrip("\n"), b.rstrip("\n")
        # if the two lines matches, simply print in white
        if la == lb:
            print(la, end="")
        else:
            # otherwise print the missmatching line two times,
            # one in green with the differing character highlighted
            for ca, cb in zip(la, lb):
                if ca == cb:
                    cprint(ca, "green", end="")
                else:
                    cprint(ca, "black", "on_green", end="")
            print()
            # one in red color
            for ca, cb in zip(la, lb):
                if ca == cb:
                    cprint(ca, "red", end="")
                else:
                    cprint(cb, "black", "on_red", end="")
        print()
