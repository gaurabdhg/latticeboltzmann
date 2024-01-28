import sys,time
from functions import * 
from simulate import run
import meshCreation as mC
import init


if __name__=="__main__":

    print("Looking for arguments passed.")
    if len(sys.argv) == 2:
        arg=sys.argv[1]
        print("For simplicity we'll now require the characteristic length of the shape you've used")
        ch=input()
        assert ch.isnumeric()

    elif len(sys.argv) == 1:     
        print('No arguments passed, will proceed with default obstacle.')
        ch=0
        arg=False
    else:
        print("Improper number of arguments")
        print("Press C to continue as without argument passed or any other key to exit.")
        aa=input()
        if aa.upper()=='C':
            arg=False
            ch=0
        else:
            print("System will now exit.")
            time.sleep(1)
            sys.exit() 

    print('\n')
    print('''This module can only read 2D images. If your file isn't formatted properly simulation will exit.''')
    
    init.defineConstants(characteristicLength=ch)
    mesh,mask= mC.defineObstacle(arg)
    run(mesh,mask)