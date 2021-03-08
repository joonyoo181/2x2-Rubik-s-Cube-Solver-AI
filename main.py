import random
import math
import time
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class Cube2x2:

    def __init__(self, state = ['B','B','B','B','O','O','O','O','R','R','R','R','Y','Y','Y','Y','W','W','W','W','G','G','G','G']):
        self.state = state

    def isSolved(self):

        face1 = self.state[:4]
        face2 = self.state[4:8]
        face3 = self.state[8:12]
        face4 = self.state[12:16]
        face5 = self.state[16:20]
        face6 = self.state[20:]

        for color in face1:
            if color != face1[0]:
                return False
        for color in face2:
            if color != face2[0]:
                return False
        for color in face3:
            if color != face3[0]:
                return False
        for color in face4:
            if color != face4[0]:
                return False
        for color in face5:
            if color != face5[0]:
                return False
        for color in face6:
            if color != face6[0]:
                return False
        return True

    def move(self, turn):

        temp_state = self.state.copy()

        if turn == 'F':
            self.state[0] = temp_state[2]
            self.state[1] = temp_state[0]
            self.state[3] = temp_state[1]
            self.state[2] = temp_state[3]
            self.state[14] = temp_state[7]
            self.state[15] = temp_state[5]
            self.state[8] = temp_state[14]
            self.state[10] = temp_state[15]
            self.state[17] = temp_state[8]
            self.state[16] = temp_state[10]
            self.state[7] = temp_state[17]
            self.state[5] = temp_state[16]

        elif turn == 'Fp':
            self.state[0] = temp_state[1]
            self.state[1] = temp_state[3]
            self.state[3] = temp_state[2]
            self.state[2] = temp_state[0]
            self.state[14] = temp_state[8]
            self.state[15] = temp_state[10]
            self.state[8] = temp_state[17]
            self.state[10] = temp_state[16]
            self.state[17] = temp_state[7]
            self.state[16] = temp_state[5]
            self.state[7] = temp_state[14]
            self.state[5] = temp_state[15]

        elif turn == 'L':
            self.state[4] = temp_state[6]
            self.state[5] = temp_state[4]
            self.state[7] = temp_state[5]
            self.state[6] = temp_state[7]
            self.state[12] = temp_state[23]
            self.state[14] = temp_state[21]
            self.state[0] = temp_state[12]
            self.state[2] = temp_state[14]
            self.state[16] = temp_state[0]
            self.state[18] = temp_state[2]
            self.state[23] = temp_state[16]
            self.state[21] = temp_state[18]

        elif turn == 'Lp':
            self.state[4] = temp_state[5]
            self.state[5] = temp_state[7]
            self.state[7] = temp_state[6]
            self.state[6] = temp_state[4]
            self.state[12] = temp_state[0]
            self.state[14] = temp_state[2]
            self.state[0] = temp_state[16]
            self.state[2] = temp_state[18]
            self.state[16] = temp_state[23]
            self.state[18] = temp_state[21]
            self.state[23] = temp_state[12]
            self.state[21] = temp_state[14]

        elif turn == 'R':
            self.state[8] = temp_state[10]
            self.state[9] = temp_state[8]
            self.state[11] = temp_state[9]
            self.state[10] = temp_state[11]
            self.state[15] = temp_state[3]
            self.state[13] = temp_state[1]
            self.state[20] = temp_state[15]
            self.state[22] = temp_state[13]
            self.state[19] = temp_state[20]
            self.state[17] = temp_state[22]
            self.state[3] = temp_state[19]
            self.state[1] = temp_state[17]

        elif turn == 'Rp':
            self.state[8] = temp_state[9]
            self.state[9] = temp_state[11]
            self.state[11] = temp_state[10]
            self.state[10] = temp_state[8]
            self.state[15] = temp_state[20]
            self.state[13] = temp_state[22]
            self.state[20] = temp_state[19]
            self.state[22] = temp_state[17]
            self.state[19] = temp_state[3]
            self.state[17] = temp_state[1]
            self.state[3] = temp_state[15]
            self.state[1] = temp_state[13]

        elif turn == 'U':
            self.state[12] = temp_state[14]
            self.state[13] = temp_state[12]
            self.state[15] = temp_state[13]
            self.state[14] = temp_state[15]
            self.state[21] = temp_state[5]
            self.state[20] = temp_state[4]
            self.state[9] = temp_state[21]
            self.state[8] = temp_state[20]
            self.state[1] = temp_state[9]
            self.state[0] = temp_state[8]
            self.state[5] = temp_state[1]
            self.state[4] = temp_state[0]

        elif turn == 'Up':
            self.state[12] = temp_state[13]
            self.state[13] = temp_state[15]
            self.state[15] = temp_state[14]
            self.state[14] = temp_state[12]
            self.state[21] = temp_state[9]
            self.state[20] = temp_state[8]
            self.state[9] = temp_state[1]
            self.state[8] = temp_state[0]
            self.state[1] = temp_state[5]
            self.state[0] = temp_state[4]
            self.state[5] = temp_state[21]
            self.state[4] = temp_state[20]

        elif turn == 'D':
            self.state[16] = temp_state[18]
            self.state[17] = temp_state[16]
            self.state[19] = temp_state[17]
            self.state[18] = temp_state[19]
            self.state[2] = temp_state[6]
            self.state[3] = temp_state[7]
            self.state[10] = temp_state[2]
            self.state[11] = temp_state[3]
            self.state[22] = temp_state[10]
            self.state[23] = temp_state[11]
            self.state[6] = temp_state[22]
            self.state[7] = temp_state[23]

        elif turn == 'Dp':
            self.state[16] = temp_state[17]
            self.state[17] = temp_state[19]
            self.state[19] = temp_state[18]
            self.state[18] = temp_state[16]
            self.state[2] = temp_state[10]
            self.state[3] = temp_state[11]
            self.state[10] = temp_state[22]
            self.state[11] = temp_state[23]
            self.state[22] = temp_state[6]
            self.state[23] = temp_state[7]
            self.state[6] = temp_state[2]
            self.state[7] = temp_state[3]

        elif turn == 'B':
            self.state[20] = temp_state[22]
            self.state[21] = temp_state[20]
            self.state[23] = temp_state[21]
            self.state[22] = temp_state[23]
            self.state[13] = temp_state[11]
            self.state[12] = temp_state[9]
            self.state[4] = temp_state[13]
            self.state[6] = temp_state[12]
            self.state[18] = temp_state[4]
            self.state[19] = temp_state[6]
            self.state[11] = temp_state[18]
            self.state[9] = temp_state[19]

        elif turn == 'Bp':
            self.state[20] = temp_state[21]
            self.state[21] = temp_state[23]
            self.state[23] = temp_state[22]
            self.state[22] = temp_state[20]
            self.state[13] = temp_state[4]
            self.state[12] = temp_state[6]
            self.state[4] = temp_state[18]
            self.state[6] = temp_state[19]
            self.state[18] = temp_state[11]
            self.state[19] = temp_state[9]
            self.state[11] = temp_state[13]
            self.state[9] = temp_state[12]

    def testSequence(self, sequence):

        temp_cube = Cube2x2(self.state.copy())  # .copy() is important!!!

        for turn in sequence:
            temp_cube.move(turn)

        return temp_cube.isSolved()

    def bruteSolve(self):

        if self.isSolved():
            return 'Already solved'

        moves = ['F', 'Fp', 'L', 'Lp', 'R', 'Rp', 'U', 'Up', 'D', 'Dp', 'B', 'Bp']

        for i1 in moves:
            if self.testSequence([i1]): return [i1]
        print('Move length 1 completed')

        for i1 in moves:
            for i2 in moves:
                if self.testSequence([i1,i2]): return [i1,i2]
        print('Move length 2 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    if self.testSequence([i1,i2,i3]): return [i1,i2,i3]
        print('Move length 3 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        if self.testSequence([i1,i2,i3,i4]): return [i1,i2,i3,i4]
        print('Move length 4 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            if self.testSequence([i1,i2,i3,i4,i5]): return [i1,i2,i3,i4,i5]
        print('Move length 5 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                if self.testSequence([i1,i2,i3,i4,i5,i6]): return [i1,i2,i3,i4,i5,i6]
        print('Move length 6 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    if self.testSequence([i1,i2,i3,i4,i5,i6,i7]): return [i1,i2,i3,i4,i5,i6,i7]
        print('Move length 7 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8]): return [i1,i2,i3,i4,i5,i6,i7,i8]
        print('Move length 8 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9]
        print('Move length 9 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10]
        print('Move length 10 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11]
        print('Move length 11 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    for i12 in moves:
                                                        if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12]
        print('Move length 12 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    for i12 in moves:
                                                        for i13 in moves:
                                                            if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13]
        print('Move length 13 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    for i12 in moves:
                                                        for i13 in moves:
                                                            for i14 in moves:
                                                                if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14]
        return 'Error: no solution found'

    def randomSolve(self):

        if self.isSolved():
            return 'Already solved'

        moves = ['F', 'Fp', 'L', 'Lp', 'R', 'Rp', 'U', 'Up', 'D', 'Dp', 'B', 'Bp']

        while True:

            rand_length = random.randint(1, 14)
            sequence = []

            for i in range(rand_length):
                sequence.append(moves[random.randint(0, 11)])

            if self.testSequence(sequence):
                return sequence

    def MCTS_solve(self, model, minutes):

        if self.isSolved():
            return 'Already solved'

        c = 1
        v = 0.1

        tree = [ [Node(self.state, P=model.predict(np.array([encodeOneHot(self.state)]))[0][1:])] ]
        current_pos = [0,0]
        current_node = tree[0][0]

        actions = ['F', 'Fp', 'L', 'Lp', 'R', 'Rp', 'U', 'Up', 'D', 'Dp', 'B', 'Bp']
        seconds = minutes * 60
        start = time.time()

        while (time.time() - start) < seconds:

            print('current_pos: ', current_pos)

            if current_node.children == None:

                print('Adding Children')
                new_children = []

                for action in actions:

                    child_cube = Cube2x2(current_node.state.copy())
                    child_cube.move(action)

                    children_index = 0

                    if current_pos[0] < len(tree) - 1:
                        children_index = len(tree[current_pos[0] + 1])

                    current_node.children = range(children_index, children_index + 12)

                    if child_cube.isSolved():
                        print('---------------------\nSOLVED CUBE ADDED\n---------------------')

                    new_children.append(Node(child_cube.state.copy(), P=model.predict(np.array([encodeOneHot(child_cube.state)]))[0][1:], parent=current_pos[1], previous_action=action))

                if current_pos[0] < len(tree) - 1:
                    tree[current_pos[0] + 1] += new_children
                else:
                    tree.append(new_children)

                max_value = -1000
                for i in range(12):
                    value = model.predict(np.array([encodeOneHot(current_node.state)]))[0][0]

                    if value > max_value:
                        max_value = value

                while current_node.parent != None:
                    action_i = actions.index(current_node.previous_action)

                    current_pos = [current_pos[0] - 1, current_node.parent]
                    current_node = tree[current_pos[0]][current_pos[1]]

                    if current_node.W[action_i] < max_value:
                        current_node.W[action_i] = max_value

                    current_node.N[action_i] += 1
                    current_node.L[action_i] -= v

            else:
                print('Selecting Action')

                action_i = 0
                max = -10000

                for i in range(len(actions)):
                    # U(a)
                    summation = 0

                    for node in tree[current_pos[0]]:
                        summation += node.N[i]

                    U = c * current_node.P[i] * (math.sqrt(summation) / (1 + current_node.N[i]) )
                    Q = current_node.W[i] - current_node.L[i]

                    print(U + Q)

                    if U + Q > max:
                        max = U + Q
                        action_i = i

                current_node.L[action_i] += v
                print('Action selected: ', actions[action_i])
                current_pos = [current_pos[0] + 1, current_node.children[action_i]]
                current_node = tree[current_pos[0]][current_pos[1]]

                if Cube2x2(current_node.state).isSolved():

                    while current_node.parent != None:
                        temp_action_i = actions.index(current_node.previous_action)
                        current_pos = [current_pos[0] - 1, current_node.parent]
                        current_node = tree[current_pos[0]][current_pos[1]]
                        current_node.N[temp_action_i] += 1

        for row in range(1, len(tree)):
            for node in range(len(tree[row])):
                if Cube2x2(tree[row][node].state).isSolved():
                    current_pos = [row, node]
                    current_node = tree[row][node]

                    solution = []
                    while current_node.previous_action != None:
                        solution.insert(0, current_node.previous_action)
                        current_pos = [current_pos[0] - 1, current_node.parent]
                        current_node = tree[current_pos[0]][current_pos[1]]

                    return solution
        return 'No solution found'


class Node:

    def __init__(self, state = ['B','B','B','B','O','O','O','O','R','R','R','R','Y','Y','Y','Y','W','W','W','W','G','G','G','G'], N=[0,0,0,0,0,0,0,0,0,0,0,0], W=[0,0,0,0,0,0,0,0,0,0,0,0], L=[0,0,0,0,0,0,0,0,0,0,0,0], P=[0,0,0,0,0,0,0,0,0,0,0,0], parent=None, children=None, previous_action=None):
        self.state = state

        self.N = N
        self.W = W
        self.L = L
        self.P = P

        self.parent = parent
        self.children = children
        self.previous_action = previous_action

def encodeOneHot(state):

    one_hot = []

    for s in state:
        if s == 'B':
            one_hot += [1,0,0,0,0,0]
        elif s == 'O':
            one_hot += [0,1,0,0,0,0]
        elif s == 'R':
            one_hot += [0,0,1,0,0,0]
        elif s == 'Y':
            one_hot += [0,0,0,1,0,0]
        elif s == 'W':
            one_hot += [0,0,0,0,1,0]
        elif s == 'G':
            one_hot += [0,0,0,0,0,1]

    return one_hot

def decodeOneHot(one_hot):

    state = ''

    start = 0
    end = 6

    while end <= len(one_hot):

        section = one_hot[start:end]

        if section == '100000':
            state += 'B'
        elif section == '010000':
            state += 'O'
        elif section == '001000':
            state += 'R'
        elif section == '000100':
            state += 'Y'
        elif section == '000010':
            state += 'W'
        elif section == '000001':
            state += 'G'

        start += 6
        end += 6

    return state

def ADI(minutes, model=None):

    if model == None:

        # Creating the value and policy network
        model = Sequential()
        model.add(Dense(50, input_dim=144, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(13, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Training Process

    moves = ['F', 'Fp', 'L', 'Lp', 'R', 'Rp', 'U', 'Up', 'D', 'Dp', 'B', 'Bp']
    seconds = minutes * 60
    start = time.time()

    total_samples = 0

    while (time.time() - start) < seconds:

        cube = Cube2x2()
        training_inputs = []

        for i in range(14):
            cube.move(moves[random.randint(0, 11)])
            training_inputs.append(Cube2x2(cube.state.copy()))

        total_samples += 14

        for i in range(len(training_inputs)):
            value_target = -100
            policy_target_move = None

            for move in moves:
                child = Cube2x2(training_inputs[i].state.copy())
                child.move(move)
                value_estimate = model.predict(np.array([encodeOneHot(child.state)]))[0][0]

                if child.isSolved():
                    value_estimate += 1
                else:
                    value_estimate -= 1

                if value_estimate > value_target:
                    value_target = value_estimate
                    policy_target_move = move

            policy_target = None
            if policy_target_move == 'F':
                policy_target = [1,0,0,0,0,0,0,0,0,0,0,0]
            elif policy_target_move == 'Fp':
                policy_target = [0,1,0,0,0,0,0,0,0,0,0,0]
            elif policy_target_move == 'L':
                policy_target = [0,0,1,0,0,0,0,0,0,0,0,0]
            elif policy_target_move == 'Lp':
                policy_target = [0,0,0,1,0,0,0,0,0,0,0,0]
            elif policy_target_move == 'R':
                policy_target = [0,0,0,0,1,0,0,0,0,0,0,0]
            elif policy_target_move == 'Rp':
                policy_target = [0,0,0,0,0,1,0,0,0,0,0,0]
            elif policy_target_move == 'U':
                policy_target = [0,0,0,0,0,0,1,0,0,0,0,0]
            elif policy_target_move == 'Up':
                policy_target = [0,0,0,0,0,0,0,1,0,0,0,0]
            elif policy_target_move == 'D':
                policy_target = [0,0,0,0,0,0,0,0,1,0,0,0]
            elif policy_target_move == 'Dp':
                policy_target = [0,0,0,0,0,0,0,0,0,1,0,0]
            elif policy_target_move == 'B':
                policy_target = [0,0,0,0,0,0,0,0,0,0,1,0]
            elif policy_target_move == 'Bp':
                policy_target = [0,0,0,0,0,0,0,0,0,0,0,1]

            model.fit(np.array([encodeOneHot(training_inputs[i].state)]), np.array([[value_target] + policy_target]), sample_weight=np.array([1/(i+1)]))

    print('Total Samples: ', total_samples)

    return model


input_state = input('Input the cube state: ')

start = time.time()

cube = Cube2x2(list(input_state))
print(cube.state)

print('\n--------AUTODIDACTIC ITERATION--------\n')
model = ADI(1)
model.save('model')

# model = tf.keras.models.load_model('model')
# ADI(1, model)

print('\n--------MONTE CARLO TREE SEARCH--------\n')
solution = cube.MCTS_solve(model, 1)

print('--------------------------\nSolution: ', solution, '\n--------------------------')

end = time.time()
print((end - start), 'seconds')
