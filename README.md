# 2x2-Rubiks-Cube-Solver-AI

## Summary
As part of a research program at California Institute of Technology, my team and I developed an Autonomous program that could solve 2x2 Rubik's Cube at any given state.
We researched and tested deep approximate value iteration, weighted A* search, Autodidactic Iteration, and Monte Carlo Tree Search => Ended up using Autodidactic Iteration, and Monte Carlo Tree Search with neural networks.

### Result
```model_old``` was trained for ~60 mins => ~26% accuracy <br />
```model_620min``` was trained for ~620 mins => 83.8% accuracy

### Credits
https://arxiv.org/pdf/1805.07470.pdf <br />
https://drive.google.com/file/d/10QQbVArmWCOT2V-F4Zf22iZ-0-ImW2VP/view?usp=sharing

## Running the Program

### Input the Cube State
Type the colors of all of the stickers as capital letters <br />
The cube faces are entered in this order: <br />
1. Front <br />
2. Left <br />
3. Right <br />
4. Up <br />
5. Down <br />
6. Back <br />
Each sequence of 4 characters represents 1 face of the cube. <br />
Ex) Solved State = 'BBBBOOOORRRRYYYYWWWWGGGG'

### Execute the Program
Run ```python main.py``` or ```python3 main.py``` depending on your Python version
