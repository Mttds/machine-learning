import numpy as np
import matplotlib.pyplot as plt

def MultiplyRow(M, row_num, row_num_multiple):
    # .copy() function is required here to keep the original matrix without any changes
    M_new = M.copy()
    M_new[row_num] = M_new[row_num] * row_num_multiple
    return M_new

def AddRows(M, row_num_1, row_num_2, row_num_1_multiple):
    M_new = M.copy()
    M_new[row_num_2] = row_num_1_multiple * M_new[row_num_1] + M_new[row_num_2]
    return M_new

def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()
    M_new[[row_num_1, row_num_2]] = M_new[[row_num_2, row_num_1]]
    return M_new

def plot_lines(M):
    x_1 = np.linspace(-10,10,100)
    x_2_line_1 = (M[0,2] - M[0,0] * x_1) / M[0,1]
    x_2_line_2 = (M[1,2] - M[1,0] * x_1) / M[1,1]
    
    _, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_1, x_2_line_1, '-', linewidth=2, color='#0075ff',
        label=f'$x_2={-M[0,0]/M[0,1]:.2f}x_1 + {M[0,2]/M[0,1]:.2f}$')
    ax.plot(x_1, x_2_line_2, '-', linewidth=2, color='#ff7300',
        label=f'$x_2={-M[1,0]/M[1,1]:.2f}x_1 + {M[1,2]/M[1,1]:.2f}$')

    A = M[:, 0:-1]
    b = M[:, -1::].flatten()
    d = np.linalg.det(A)

    if d != 0:
        solution = np.linalg.solve(A,b) 
        ax.plot(solution[0], solution[1], '-o', mfc='none', 
            markersize=10, markeredgecolor='#ff0000', markeredgewidth=2)
        ax.text(solution[0]-0.25, solution[1]+0.75, f'$(${solution[0]:.0f}$,{solution[1]:.0f})$', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))

    plt.xlabel('$x_1$', size=14)
    plt.ylabel('$x_2$', size=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.axis([-10, 10, -10, 10])

    plt.grid()
    plt.gca().set_aspect("equal")

    plt.show()

A = np.array(
    [
        [-1, 3],
        [3, 2]
    ], dtype=np.dtype(float)
)

b = np.array([7, 1], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)

print(f"Shape of A: {A.shape}")
print(f"Shape of b: {b.shape}")

x = np.linalg.solve(A, b)
print(f"Solution: {x}")

# non 0 determinant means non-singular matrix (exactly one solution)
d = np.linalg.det(A)
print(f"Determinant of matrix A: {d:.2f}")

# stack matrix A and array b in a single matrix
# to stack it with the (2,2) matrix we need to use .reshape((2, 1)):
A_system = np.hstack((A, b.reshape((2, 1))))
print(A_system)
print(A_system[1])

# Function .copy() is used to keep the original matrix without any changes.
A_system_res = A_system.copy()
# Multiply first equation (row) of the matrix (system) by 3 and add to the second equation
# Finally substitute the result of this operation with the second equation in the system
A_system_res[1] = 3 * A_system_res[0] + A_system_res[1]
print(A_system_res)
# Multiply this new equation by 1/11 to get the value 0x1 + 1x2 = 2 => x2 = 2
# Then the solution for x1 can be found with the first equation (row) using x2 = 2
A_system_res[1] = 1/11 * A_system_res[1]
print(A_system_res)

plot_lines(A_system)

A = np.array([
        [4, -3, 1],
        [2, 1, 3],
        [-1, 2, -5]
    ], dtype=np.dtype(float))

b = np.array([-10, 0, 17], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)

print(f"Shape of A: {np.shape(A)}")
print(f"Shape of b: {np.shape(b)}")

x = np.linalg.solve(A, b)
print(f"Solution: {x}")

d = np.linalg.det(A)
print(f"Determinant of matrix A: {d:.2f}")

A_system = np.hstack((A, b.reshape((3, 1))))
print(A_system)

print("Original matrix:")
print(A_system)
print("\nMatrix after its third row is multiplied by 2:")
print(MultiplyRow(A_system,2,2))

print("Original matrix:")
print(A_system)
print("\nMatrix after exchange of the third row with the sum of itself and second row multiplied by 1/2:")
print(AddRows(A_system,1,2,1/2))

print("Original matrix:")
print(A_system)
print("\nMatrix after exchange its first and third rows:")
print(SwapRows(A_system,0,2))

A_ref = SwapRows(A_system,0,2)
# Note: ref is an abbreviation of the row echelon form (row reduced form)
print(A_ref)
# multiply row 0 of the new matrix A_ref by 2 and add it to the row 1
A_ref = AddRows(A_ref,0,1,2)
print(A_ref)
# multiply row 0 of the new matrix A_ref by 4 and add it to the row 2
A_ref = AddRows(A_ref,0,2,4)
print(A_ref)
# multiply row 1 of the new matrix A_ref by -1 and add it to the row 2
A_ref = AddRows(A_ref,1,2,-1)
print(A_ref)
# multiply row 2 of the new matrix A_ref by -1/12
A_ref = MultiplyRow(A_ref,2,-1/12)
print(A_ref)
x_3 = -2
x_2 = (A_ref[1,3] - A_ref[1,2] * x_3) / A_ref[1,1]
x_1 = (A_ref[0,3] - A_ref[0,2] * x_3 - A_ref[0,1] * x_2) / A_ref[0,0]

print(x_1, x_2, x_3)
