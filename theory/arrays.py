import numpy as np

# 1d array
a = np.array([10,12])
print(a)

# create array that starts at 0 and ends at 3 (incrementing by 1)
a = np.arange(3) # array of 3 elements
print(a)

# create array that starts at from 1, ends at 20, increments by 3
a = np.arange(1, 20, 3)
print(a)

# linspace is array with 5 evenly spaced values from 0 to 100
# arange is array with n 100/5 values from 0 to 100-5
a = np.arange(0, 100, 5)
b = np.linspace(0, 100, 5)
print(a)
print(b)

# same as before but with int datatype for numpy
b = np.linspace(0, 100, 5, dtype=int)
print(b)

# array of ones
a = np.ones(3)
print(a)

# array of zeroes
a = np.zeros(3)
print(a)

# empty array
a = np.empty(3)
print(a)

# random array of 3 entries (between 0 and 1 float)
a = np.random.rand(3)
print(a)

# 2d array
A = np.array([[1,2,3], [4,5,6]])
print(A)

# 1-D array 
a = np.array([1, 2, 3, 4, 5, 6])

# Multidimensional array using reshape()
A = np.reshape(
    a, # the array to be reshaped
    (2,3) # dimensions of the new array
)
# Print the new 2-D array with two rows and three columns
print(a)
print(A)

print(A.ndim, A.size, A.shape)

# array ops
arr_1 = np.array([2, 4, 6])
arr_2 = np.array([1, 3, 5])

# Adding two 1-D arrays
addition = arr_1 + arr_2
print(addition)

# Subtracting two 1-D arrays
subtraction = arr_1 - arr_2
print(subtraction)

# Multiplying two 1-D arrays elementwise
multiplication = arr_1 * arr_2
print(multiplication)

# scalar mult (broadcasting)
vector = np.array([1, 2])
vector = vector * 1.6
print(vector)

# indexing and slicing
# Select the third element of the array. Remember the counting starts from 0.
a = ([1, 2, 3, 4, 5])
print(a[2])

# Select the first element of the array.
print(a[0])

# Indexing on a 2-D array
two_dim = np.array(([1, 2, 3],
          [4, 5, 6], 
          [7, 8, 9]))

# Select element number 8 from the 2-D array using indices i, j.
print(two_dim[2][1])

# Slice the array a to get the array [2,3,4]
sliced_arr = a[1:4]
print(sliced_arr)

# Slice the array a to get the array [1,2,3]
sliced_arr = a[:3]
print(sliced_arr)

# Slice the array a to get the array [1,3,5]
sliced_arr = a[::2]
print(sliced_arr)

# Note that a == a[:] == a[::]
print(a == a[:] == a[::])

# Slice the two_dim array to get the first two rows
sliced_arr_1 = two_dim[0:2]
print(sliced_arr_1)

# Similarily, slice the two_dim array to get the last two rows
sliced_two_dim_rows = two_dim[1:3]
print(sliced_two_dim_rows)

sliced_two_dim_cols = two_dim[:,1]
print(sliced_two_dim_cols)

# stacking arrays (joining)
a1 = np.array([[1,1], 
               [2,2]])
a2 = np.array([[3,3],
              [4,4]])
print(f'a1:\n{a1}')
print(f'a2:\n{a2}')

# Stack the arrays vertically
vert_stack = np.vstack((a1, a2))
print(vert_stack)

# Stack the arrays horizontally
horz_stack = np.hstack((a1, a2))
print(horz_stack)
