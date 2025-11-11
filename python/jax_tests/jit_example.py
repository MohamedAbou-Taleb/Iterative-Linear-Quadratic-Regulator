import jax
import jax.numpy as jnp
import timeit

# 1. Define a function that operates on a large vector
#    but returns a single number (e.g., a loss function).
def loss_function(x):
  """A simple 'loss' is just the sum of our polynomial."""
  return jnp.sum(x**3 - 2*x + 1)

# 2. Get the gradient function.
#    The gradient will be a vector of the same shape as x.
grad_loss = jax.grad(loss_function)

# 3. Create a JIT-compiled version of the gradient function.
#    This is the key step!
jit_grad_loss = jax.jit(grad_loss)

# 4. Create large sample data
key = jax.random.PRNGKey(0)
large_x = jax.random.normal(key, (1000000,))

# --- Time the NON-JIT gradient ---
def time_original_grad():
  grad_loss(large_x).block_until_ready()

original_time = timeit.timeit(time_original_grad, number=100)
print(f"\n--- Example 2: Speedup ---")
print(f"Gradient without JIT: {original_time / 100:.6f} seconds (average over 100 runs)")

# --- Time the JIT gradient ---

# "Warm up" the JIT cache (pay the one-time compilation cost)
print("Running JIT compilation (warm-up)...")
jit_grad_loss(large_x).block_until_ready()
print("Warm-up complete.")

# Now, time the fast, compiled gradient function
def time_jit_grad():
  jit_grad_loss(large_x).block_until_ready()

jit_time = timeit.timeit(time_jit_grad, number=100)
print(f"Gradient with JIT:    {jit_time / 100:.6f} seconds (average over 100 runs)")

# --- Show the results ---
speedup = original_time / jit_time
print(f"\nSpeedup: {speedup:.2f}x")