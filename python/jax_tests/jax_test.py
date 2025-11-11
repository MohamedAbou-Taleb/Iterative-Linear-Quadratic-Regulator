import jax
from jax import numpy as jnp


# --- 1. Setup ---
# JAX arrays are the basic building block.
# They are immutable, meaning you can't change them in-place.
print("--- 1. Setup ---")
key = jax.random.PRNGKey(0)  # JAX requires an explicit random key
a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])

print(f"vector a: {a}")
print(f"Vector b: {b}")

print("\n--- 2. Basic Operations ---")

# Addition
add_result = a + b
print(f"Addition (a + b): {add_result}")
# Multiplication (element-wise)
mul_result = a * b
print(f"Multiplication (a * b): {mul_result}")

# --- 3. Linear Algebra ---
print("\n--- 3. Linear Algebra ---")

# Inner Product (Dot Product)
# This is a key operation in many ML/scientific models
inner_prod = jnp.dot(a, b)
inner_prod = a@b
print(f"Inner Product (jnp.dot(a, b)): {inner_prod}")

# Create two matrices
mat_A = jax.random.normal(key, (2, 3)) # 2x3 matrix
mat_B = jax.random.normal(key, (3, 2)) # 3x2 matrix

print(f"\nMatrix A (2x3):\n{mat_A}")
print(f"Matrix B (3x2):\n{mat_B}")

# Matrix Multiplication
# You can use jnp.matmul() or the standard Python '@' operator
mat_prod = mat_A @ mat_B
# or: mat_prod = jnp.matmul(mat_A, mat_B)
print(f"Matrix Multiplication (A @ B):\n{mat_prod}")

# Define a simple Python function that does math.
# Let's use f(x) = x^3 + 2x + 1
def my_function(x):
  return x**3 + 2.0 * x + 1.0

# The derivative is f'(x) = 3x^2 + 2

# Use jax.grad() to create a NEW function that computes the derivative
grad_f = jax.grad(my_function)
# Now, let's test it at a specific point, say x = 3.0
x_val = 3.0
derivative_at_3 = grad_f(x_val)

# Check our math: 3*(3.0**2) + 2 = 3*9 + 2 = 27 + 2 = 29.0
print(f"Original function f(x) = x^3 + 2x + 1")
print(f"Value of f(3.0): {my_function(x_val)}")
print(f"Value of f'(3.0): {derivative_at_3}")
print(f"Expected derivative: 29.0")

# jax.grad also works with functions that take multiple arguments
def complex_func(x, y):
  return x**2 * y + jnp.sin(x)

# By default, grad differentiates with respect to the FIRST argument (arg=0)
grad_df_dx = jax.grad(complex_func) 
# Differentiate with respect to the SECOND argument (arg=1)
grad_df_dy = jax.grad(complex_func, argnums=1)

x_val = 2.0
y_val = 3.0
print(f"\nComplex function g(x, y) = x^2 * y + sin(x)")
print(f"Derivative w.r.t. x at (2, 3): {grad_df_dx(x_val, y_val)}")
print(f"Derivative w.r.t. y at (2, 3): {grad_df_dy(x_val, y_val)}")